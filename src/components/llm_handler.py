import ollama
from typing import Dict, Any, Generator
import random
import os
import sys
import traceback

# RAG Imports
from minirag import MiniRAG, QueryParam
from minirag.llm.ollama import ollama_model_complete
from indexer import *

class LLMHandler:
    def __init__(self, config=None):
        """Initialize LLM Handler, including RAG querier if enabled."""
        config = config or {}
        self.model_name = config.get('model', 'gemma3:4b') # Get base model from main config
        
        # Text generation parameters from main config
        local_config = config.get('local', {})
        self.params = {
            'temperature': local_config.get('temperature'),
            'top_p': local_config.get('top_p'),
            'top_k': local_config.get('top_k'),
            'max_tokens': local_config.get('max_tokens'),
            'n_ctx': local_config.get('n_ctx'),
            'repeat_penalty': local_config.get('repeat_penalty')
        }
        print(f"LLM Handler initialized for base model: {self.model_name}")

        # --- RAG Initialization ---
        self.rag_querier = None
        enable_rag_str = os.getenv('ENABLE_RAG', 'false')
        cleaned_enable_rag_str = enable_rag_str.split('#')[0].strip().strip('"').strip("'").lower()
        self.rag_enabled = cleaned_enable_rag_str == 'true'

        if self.rag_enabled:
            print("RAG is enabled. Attempting to initialize MiniRAG querier...")
            # Load RAG-specific config from environment
            self.working_dir = os.getenv('WORKING_DIR')
            raw_query_llm_model = os.getenv('QUERY_LLM_MODEL')
            self.embedding_model = os.getenv('EMBEDDING_MODEL')
            llm_max_token = int(os.getenv('LLM_MAX_TOKEN_SIZE', '200'))
            llm_max_async = int(os.getenv('LLM_MAX_ASYNC', '1'))
            
            # Clean the query model name
            self.query_llm_model = None
            if raw_query_llm_model:
                 self.query_llm_model = raw_query_llm_model.split('#')[0].strip().strip('"').strip("'")
                 print(f"Cleaned QUERY_LLM_MODEL: '{self.query_llm_model}' (from '{raw_query_llm_model}')")

            required_rag_vars = {'WORKING_DIR': self.working_dir, 
                                 'QUERY_LLM_MODEL': self.query_llm_model, 
                                 'EMBEDDING_MODEL': self.embedding_model}
            missing_vars = [name for name, value in required_rag_vars.items() if not value]
            
            if missing_vars:
                print(f"Warning: Cannot initialize RAG. Missing env vars: {missing_vars}. RAG disabled.")
                self.rag_enabled = False
            else:
                rag_embedding_func = setup_embedding_func(self.embedding_model)
                
                if rag_embedding_func:
                    try:
                        self.rag_querier = MiniRAG(
                            working_dir=self.working_dir,
                            llm_model_func=ollama_model_complete, # Use Ollama for querying
                            llm_model_max_token_size=llm_max_token, # Use RAG settings
                            llm_model_max_async=llm_max_async,
                            llm_model_kwargs={"ollama_model": self.query_llm_model},
                            embedding_func=rag_embedding_func,
                        )
                    except Exception as e:
                        print(f"Error initializing MiniRAG Querier: {e}")
                        traceback.print_exc()
                        print("RAG disabled due to MiniRAG initialization error.")
                        self.rag_querier = None
                        self.rag_enabled = False
                else:
                     print("RAG disabled due to embedding function setup error.")
                     self.rag_enabled = False
        else:
            print("RAG is disabled via ENABLE_RAG environment variable.")
        
    def get_response(self, messages: list[Dict[str, Any]]) -> Generator[str, None, None]:
        """Get a streaming response from the base LLM."""
        print(f"Using Base LLM '{self.model_name}' with params: {self.params}")
        try:
            response = ollama.chat(
                model=self.model_name, 
                messages=messages,
                stream=True,
                options=self.params
            )
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
             print(f"\nError during Ollama chat with base model: {e}")
             traceback.print_exc()
             yield f"[Error communicating with base LLM: {e}]" # Yield error message
                
    def get_rag_response(self, query: str, messages: list[Dict[str, Any]]) -> Generator[str, None, None]:
        """Get a response using the MiniRAG querier instance, falling back to base LLM if no context found."""
        if not self.rag_querier:
            print("Error: get_rag_response called but RAG querier is not available.")
            yield "[Internal Error: RAG is not available]"
            return
        
        # Extract system prompt from messages history (assuming it's the first message)
        system_prompt = "You are a helpful assistant." # Default fallback
        if messages and messages[0]["role"] == "system":
             system_prompt = messages[0]["content"]
             print(f"[Debug RAG] Extracted system prompt for RAG query: '{system_prompt[:60]}...'")
        else:
             print("[Debug RAG] No system prompt found in messages, using default.")

        # Create QueryParam with the extracted system prompt
        rag_param = QueryParam(mode="naive", system_prompt=system_prompt)
            
        print(f"Using RAG Querier (LLM: {self.query_llm_model}) with mode '{rag_param.mode}'")
        try:
            answer = self.rag_querier.query(
                query, 
                param=rag_param
            )
            
            # ---> Reintroduce Fallback Logic <--- 
            if answer is None:
                 print("RAG query returned None (no context found). Falling back to base LLM.")
                 rag_failure_note = {
                     "role": "system", 
                     "content": "(Self-Correction: My knowledge base didn\'t have specific information for that query. Answering from general knowledge.)"
                 }
                 modified_messages = messages + [rag_failure_note]
                 # Call the standard get_response with the modified history
                 yield from self.get_response(messages=modified_messages)
                 return # Stop this generator
            # ---> End Fallback Logic <--- 

            # RAG query succeeded, yield the answer char by char (as reverted)
            print("RAG query successful. Yielding answer character by character...")
            if answer:
                 for char in answer:
                     yield char
            else:
                 # Handle case where RAG returns empty string (not None)
                 yield "[RAG query returned an empty answer]"
                
        except Exception as e:
            print(f"\nError during MiniRAG query execution: {e}")
            traceback.print_exc()
            yield f"[Error during RAG query: {e}]"