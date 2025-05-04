import ollama
from typing import Dict, Any, Generator, AsyncIterator, Optional
import random
import os
import sys
import traceback
from datetime import datetime, timedelta
import re
import pathlib # Added for path manipulation
from minirag import MiniRAG, QueryParam
from minirag.llm.ollama import ollama_model_complete
from indexer import *
from config.personality_config import PERSONALITY_CORE
from minirag.prompt import PROMPTS
from utils.conversation_manager import format_timedelta

CONTEXT_LENGTH_LIMIT = int(os.getenv('CONTEXT_LENGTH_LIMIT', '5000'))

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
        self.base_data_path = os.getenv('DATA_PATH')
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
        
    def _get_dynamic_context(self) -> dict:
        """Loads the latest summary and calculates time since last interaction."""
        context = {"time_since_last": "N/A (First interaction likely)", "conversation_summary": "N/A (No previous summary found)"}
        if not self.base_data_path:
            return context # Return default if path is not set

        summaries_dir = pathlib.Path(self.base_data_path) / "summaries"
        if not summaries_dir.is_dir():
            return context # Return default if dir doesn't exist

        try:
            summary_files = list(summaries_dir.glob("summary_*.txt"))
            if not summary_files:
                return context # Return default if no summary files found

            latest_summary_file = max(summary_files, key=lambda p: p.stat().st_mtime)

            # Extract timestamp from filename
            match = re.search(r'summary_(\d{8}_\d{6})\.txt', latest_summary_file.name)
            if match:
                timestamp_str = match.group(1)
                last_interaction_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                time_delta = datetime.now() - last_interaction_time
                context["time_since_last"] = format_timedelta(time_delta)

            # Read summary content (skip metadata lines)
            with open(latest_summary_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Correctly join lines after the metadata (first 3 lines)
                summary_content = "".join(lines[3:])
                context["conversation_summary"] = summary_content.strip() if summary_content else "N/A (Summary file was empty)"

        except Exception as e:
            print(f"Error loading dynamic context: {e}")
            traceback.print_exc()
            # Keep default context values on error

        return context

    def get_response(self, messages: list[Dict[str, Any]], rag_context: Optional[str] = None) -> Generator[str, None, None]:
        """Get a streaming response from the base LLM, injecting personality and optional RAG context within a single system prompt."""
        print(f"Using Base LLM '{self.model_name}' with params: {self.params}")
        
        dynamic_context = self._get_dynamic_context()
        rag_context_for_prompt = "None." # Default if no context

        # --- Temporarily disable RAG context injection for testing --- #
        if rag_context and rag_context != PROMPTS["fail_response"]:
            if len(rag_context) > CONTEXT_LENGTH_LIMIT:
                rag_context_for_prompt = rag_context[:CONTEXT_LENGTH_LIMIT] + "... (truncated)"
            else:
                rag_context_for_prompt = rag_context
        else:
            print("[Debug LLMHandler] No valid RAG context found or provided.")

        formatted_personality = PERSONALITY_CORE.format(
            time_since_last=dynamic_context.get("time_since_last", "N/A"),
            conversation_summary=dynamic_context.get("conversation_summary", "N/A"),
            rag_context=rag_context_for_prompt
        )

        modified_messages = []
        temp_messages = [m for m in messages if m['role'] != 'system']
        modified_messages.append({'role': 'system', 'content': formatted_personality})
        modified_messages.extend(temp_messages)

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=modified_messages,
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
                
    async def get_rag_response(self, query: str, messages: list[Dict[str, Any]]) -> Generator[str, None, None]:
        """Uses MiniRAG to retrieve context based *only* on the latest query, then calls get_response to generate the final answer."""
        if not self.rag_querier:
            return self.get_response(messages=messages, rag_context=None)

        try:
            if not self.rag_querier:
                 raise ValueError("RAG querier is not initialized.")
            
            print("Attempting RAG query...")
            # Call the RAG query method - Assuming it returns context string or similar
            # We now need to await the async query method
            context_result = await self.rag_querier.query(
                query=query, 
                param=QueryParam(
                    mode="mini", 
                    only_need_context=True
                )
            )
            print(f"RAG Query Result Type: {type(context_result)}")

            # Check if the result is usable context (adjust based on actual return type)
            # If query returns a generator, we need to consume it here to get the context.
            # Assuming for now query returns a string context or None/empty string on failure.
            context_str = ""
            if isinstance(context_result, str):
                 context_str = context_result
            elif hasattr(context_result, '__aiter__'): # Check if it's an async iterator
                 print("Consuming RAG async generator result...")
                 context_str = "".join([chunk async for chunk in context_result])
            elif hasattr(context_result, '__iter__'): # Check if it's a sync iterator
                 print("Consuming RAG sync generator result...")
                 context_str = "".join(list(context_result))

            if context_str and context_str.strip():
                rag_context = context_str.strip()
                print(f"RAG Context Retrieved ({len(rag_context)} chars): '{rag_context[:100]}...'")
            else:
                print("RAG query returned no usable context.")
                rag_context = None # Ensure it's None if empty or invalid
                
        except ValueError as ve:
             print(f"RAG configuration error: {ve}")
        except Exception as e:
            print(f"\nError during RAG context retrieval: {e}")
            traceback.print_exc()
            # Fall through to base LLM call if RAG fails
            
        # --- Prepare messages for final LLM call --- 

        return self.get_response(messages=messages, rag_context=rag_context)