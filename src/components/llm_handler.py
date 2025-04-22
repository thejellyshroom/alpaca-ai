import ollama
from typing import Dict, Any, Generator, AsyncIterator, Optional
import random
import os
import sys
import traceback
from datetime import datetime, timedelta
import re
import pathlib # Added for path manipulation

# RAG Imports
from minirag import MiniRAG, QueryParam
from minirag.llm.ollama import ollama_model_complete
from indexer import *

# Personality Import
from config.personality_config import PERSONALITY_CORE
from minirag.prompt import PROMPTS

# --- Helper Function for time formatting ---
def _format_timedelta(delta: timedelta) -> str:
    """Formats timedelta into human-readable string (e.g., '5 minutes ago')."""
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds} seconds ago"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        days = seconds // 86400
        return f"{days} day{'s' if days > 1 else ''} ago"
# --- End Helper ---

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
                context["time_since_last"] = _format_timedelta(time_delta)

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

        # Prepare RAG context string for formatting
        rag_context_for_prompt = "None." # Default if no context
        if rag_context and rag_context != PROMPTS["fail_response"]:
            print("[Debug LLMHandler] Valid RAG context found.")
            rag_context_for_prompt = rag_context
        else:
            print("[Debug LLMHandler] No valid RAG context found or provided.")

        # Format the single personality prompt with all context
        formatted_personality = PERSONALITY_CORE.format(
            time_since_last=dynamic_context["time_since_last"],
            conversation_summary=dynamic_context["conversation_summary"],
            rag_context=rag_context_for_prompt # Inject RAG context here
        )

        # Prepare final messages list
        modified_messages = []
        # Ensure system prompt is the first message
        temp_messages = [m for m in messages if m['role'] != 'system'] # Remove existing system prompts
        modified_messages.append({'role': 'system', 'content': formatted_personality})

        # Add the rest of the non-system messages
        modified_messages.extend(temp_messages)

        print(f"[Debug Personality] Using single dynamic system prompt: {modified_messages[0]['content'][:300]}...") # Increased length for debug

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=modified_messages, # Use modified messages
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
                
    def _format_history_for_rag_query(self, messages: list[Dict[str, Any]], num_turns=2) -> str:
        """Formats the last N turns of conversation for RAG query context."""
        # Find the system prompt if it exists
        system_prompt = ""
        for msg in messages:
            if msg['role'] == 'system':
                # Use the *formatted* system prompt if available (contains dynamic context)
                system_prompt = msg['content']
                break
            
        # Get last N user/assistant messages (2*num_turns messages)
        user_assistant_messages = [m for m in messages if m['role'] in ('user', 'assistant')]
        recent_messages = user_assistant_messages[-(2*num_turns):]
        
        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages])
        
        # Combine system prompt (if any) and recent history
        # Avoid adding system prompt if it wasn't found or is just the base unformatted one
        if system_prompt and system_prompt.strip().startswith("You are Alpaca"):
             # Find the original query (last user message)
             last_user_message = recent_messages[-1]['content'] if recent_messages and recent_messages[-1]['role'] == 'user' else ""
             # Return history excluding the last user message, which will be appended as the main query
             history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages[:-1]])
             return f"Relevant Conversation History:\n{history_context}\n---" # Separator
        else:
             # Fallback if system prompt isn't formatted or found - just use recent turns
             return f"Relevant Conversation History:\n{history_str}\n---" # Separator

    async def get_rag_response(self, query: str, messages: list[Dict[str, Any]]) -> AsyncIterator[str]:
        """Uses MiniRAG to retrieve context based on query + history, then calls get_response to generate the final answer."""
        if not self.rag_querier:
            print("Skipping RAG: Querier not available. Calling base LLM directly.")
            # Fallback to base LLM directly without RAG context
            return self.get_response(messages=messages, rag_context=None)

        # --- Prepare for RAG Context Retrieval --- #
        rag_param = QueryParam(mode="naive", only_need_context=True) # Set mode and request only context

        # --- Construct combined query with history --- #
        # We only need the message history here, not the fully formatted personality
        # The dynamic personality formatting happens later in get_response
        # Use the original messages list for history formatting
        history_context_str = self._format_history_for_rag_query(messages, num_turns=2)
        # Corrected f-string formatting
        combined_query = f"{history_context_str}\nUser Query: {query}"
        # --- End Construct Query --- #

        print(f"Attempting RAG context retrieval (mode: {rag_param.mode}) with combined query...")
        retrieved_context = None
        try:
            # Call aquery to get *only* the context string
            context_result = await self.rag_querier.aquery(
                combined_query,
                param=rag_param # Pass param with only_need_context=True
            )

            if isinstance(context_result, str) and context_result != PROMPTS["fail_response"]:
                print(f"RAG retrieval successful. Context length: {len(context_result)}")
                retrieved_context = context_result
            elif context_result is None:
                 print("RAG retrieval returned None (no context found).")
            else:
                 print(f"RAG retrieval failed or returned fail response: {context_result}")

        except Exception as e:
            print(f"\nError during RAG context retrieval: {e}")
            traceback.print_exc()
            retrieved_context = None # Ensure context is None on error
        # --- End RAG Context Retrieval --- #

        # --- Call Base LLM with or without RAG context --- #
        # get_response will handle dynamic personality injection and adding the RAG context message if available.
        print("Calling get_response to generate final answer...")
        return self.get_response(messages=messages, rag_context=retrieved_context)