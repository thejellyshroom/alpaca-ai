import ollama
from typing import Dict, Any, Generator, AsyncIterator
import random
import os
import sys
import traceback

# RAG Imports
from minirag import MiniRAG, QueryParam
from minirag.llm.ollama import ollama_model_complete
from indexer import *

# Personality Import
from src.config.personality_config import PERSONALITY_CORE

class LLMHandler:
    def __init__(self, config=None):
        """Initialize LLM Handler, including RAG querier if enabled."""
        config = config or {}
        self.model_name = config.get('model', 'gemma3:4b')
        local_config = config.get('local', {})
        self.params = {
            'temperature': local_config.get('temperature'),
            'top_p': local_config.get('top_p'),
            'top_k': local_config.get('top_k'),
            'max_tokens': local_config.get('max_tokens'),
            'n_ctx': local_config.get('n_ctx'),
            'repeat_penalty': local_config.get('repeat_penalty')
        }
        self.rag_querier = None
        enable_rag_str = os.getenv('ENABLE_RAG', 'false')
        cleaned_enable_rag_str = enable_rag_str.split('#')[0].strip().strip('"').strip("'").lower()
        self.rag_enabled = cleaned_enable_rag_str == 'true'
        if self.rag_enabled:
            self.working_dir = os.getenv('WORKING_DIR')
            raw_query_llm_model = os.getenv('QUERY_LLM_MODEL')
            self.embedding_model = os.getenv('EMBEDDING_MODEL')
            llm_max_token = int(os.getenv('LLM_MAX_TOKEN_SIZE', '200'))
            llm_max_async = int(os.getenv('LLM_MAX_ASYNC', '1'))
            self.query_llm_model = None
            if raw_query_llm_model:
                 self.query_llm_model = raw_query_llm_model.split('#')[0].strip().strip('"').strip("'")
                 print(f"QUERY_LLM_MODEL: {self.query_llm_model}")

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
                            llm_model_func=ollama_model_complete,
                            llm_model_max_token_size=llm_max_token,
                            llm_model_max_async=llm_max_async,
                            llm_model_kwargs={"ollama_model": self.query_llm_model},
                            embedding_func=rag_embedding_func,
                            # Inject personality core into MiniRAG's global config
                            global_config={"personality_core": PERSONALITY_CORE} 
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
        """Get a streaming response from the base LLM, injecting personality."""
        print(f"Using Base LLM '{self.model_name}' with params: {self.params}")
        
        system_message_found = False
        for msg in messages:
            if msg['role'] == 'system':
                # Prepend personality to existing system message
                original_content = msg.get('content', '')
                msg['content'] = f"{PERSONALITY_CORE}\n\n{original_content}"
                system_message_found = True
                break
        
        if not system_message_found:
            # Insert personality as the system message if none exists
            messages.insert(0, {'role': 'system', 'content': PERSONALITY_CORE})
            
        print(f"[Debug Personality] System prompt for base LLM: {messages[0]['content'][:100]}...")
        # --- End Personality Injection --- 
            
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
                
    async def get_rag_response(self, query: str, messages: list[Dict[str, Any]]) -> AsyncIterator[str]:
        """Get a streaming response using the MiniRAG querier instance, falling back to base LLM if no context found."""
        if not self.rag_querier:
            print("Error: get_rag_response called but RAG querier is not available.")
            async def error_gen(): 
                yield "[Internal Error: RAG is not available]"
            return error_gen()
        
        rag_param = QueryParam(mode="naive") 
            
        print(f"Using RAG Querier (LLM: {self.query_llm_model}) with mode '{rag_param.mode}'")
        try:
            answer_source = await self.rag_querier.aquery(
                query, 
                param=rag_param
            )
            
            # ---> Fallback Logic with Personality <--- 
            if answer_source is None or isinstance(answer_source, str): # Handle None or error string from RAG
                 fallback_reason = "no context found" if answer_source is None else f"RAG error: {answer_source}"
                 print(f"RAG query failed ({fallback_reason}). Falling back to base LLM with personality.")
                 rag_failure_note = {
                     "role": "system", 
                     "content": f"{PERSONALITY_CORE}\n\n(Self-Correction: My knowledge base lookup failed ({fallback_reason}). I'll answer from general knowledge, but don't expect miracles. 🙄)"
                 }
                 non_system_messages = [m for m in messages if m['role'] != 'system']
                 modified_messages = [rag_failure_note] + non_system_messages
                 
                 # Return the generator from the base LLM call (get_response handles streaming)
                 return self.get_response(messages=modified_messages) 
            # ---> End Fallback Logic <--- 

            # --- Return the Async Generator Directly --- 
            # RAG query succeeded, answer_source is the AsyncIterator from aquery/ollama
            print("RAG query successful. Returning response stream...")
            return answer_source # Return the generator directly
                
        except Exception as e:
            print(f"\nError during MiniRAG query invocation: {e}")
            traceback.print_exc()
            # --- Fix NameError by capturing error message --- 
            error_message = f"[Error during RAG query invocation: {e}]"
            async def error_gen(): 
                 yield error_message 
            return error_gen()