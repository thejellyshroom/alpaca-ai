# src/utils/session_utils.py

import asyncio
from datetime import datetime
import os
import traceback

# Need to import the specific handlers/managers
# Adjust paths based on actual project structure relative to this file
# Assuming this file is in src/utils, components are in src/components
from ..components.llm_handler import LLMHandler 
from .conversation_manager import ConversationManager 

# Define the prompt for the summarization task
SUMMARY_PROMPT_TEMPLATE = """
You are an objective summarization assistant.
Given the following conversation history between a User and an AI Assistant (Alpaca), provide a concise summary focusing on:
1. The main questions or topics initiated by the User.
2. The key information, conclusions, or actions provided by the Assistant.
3. Any notable shifts in topic or recurring themes.
Keep the summary factual and neutral. Do not add any introductory or concluding remarks like \"Here is the summary:\".

Conversation History:
{history_string}

Concise Summary:
"""

# Define the dedicated system prompt for the summarizer persona
SUMMARIZER_SYSTEM_PROMPT = {"role": "system", "content": "You are an objective summarization assistant tasked with creating a summary of a conversation. The summary should be able to capture the intricacies of the conversation and the key points that were discussed."}

def _format_history_for_prompt(history: list[dict]) -> str:
    """Formats conversation history into a readable string."""
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

async def summarize_conversation(history: list[dict], llm_handler: LLMHandler) -> str:
    """Generates a summary of the conversation history using the LLM."""
    if not history:
        print("[Summarizer] History is empty, skipping summarization.")
        return ""

    # Filter out system messages before summarizing user/assistant interaction
    user_assistant_history = [msg for msg in history if msg['role'] in ('user', 'assistant')]
    if not user_assistant_history:
        print("[Summarizer] No user/assistant messages found in history, skipping summarization.")
        return ""

    print("[Summarizer] Formatting history for summarization...")
    history_string = _format_history_for_prompt(user_assistant_history)
    prompt = SUMMARY_PROMPT_TEMPLATE.format(history_string=history_string)

    # Prepare messages for the specific summarization call
    summarization_messages = [
        SUMMARIZER_SYSTEM_PROMPT,
        {"role": "user", "content": prompt}
    ]

    print("[Summarizer] Requesting summary from LLM...")
    try:
        # Use the LLM Handler's get_response, which returns a sync generator
        summary_generator = llm_handler.get_response(messages=summarization_messages)
        
        # Consume the SYNCHRONOUS generator fully using a standard for loop
        chunks = []
        for chunk in summary_generator:
             chunks.append(chunk)
        full_summary = "".join(chunks)
             
        print("[Summarizer] Summary received.")
        return full_summary.strip()
    except Exception as e:
        print(f"[Summarizer] Error during LLM call for summarization: {e}")
        traceback.print_exc() # Log traceback for debugging
        return "[Error generating summary]" # Return error indicator

def save_summary(summary: str, history_length: int, base_data_path: str):
    """Saves the summary text and metadata to a timestamped file within the specified base data path."""
    if not summary or summary == "[Error generating summary]":
        print("[Summarizer] Skipping save due to empty or error summary.")
        return
        
    if not base_data_path:
        print("[Summarizer] Error: Base data path not provided. Cannot save summary.")
        return

    try:
        # Construct the specific summaries directory path
        summaries_dir = os.path.join(base_data_path, "summaries") 
        
        os.makedirs(summaries_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(summaries_dir, f"summary_{timestamp_str}.txt")

        metadata = f"Timestamp: {datetime.now().isoformat()}\nTurns: {history_length}\n---\n"
        content_to_write = metadata + summary

        print(f"[Summarizer] Saving summary to {filename}...")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content_to_write)
        print("[Summarizer] Summary saved successfully.")

    except IOError as e:
        print(f"[Summarizer] Error saving summary file {filename}: {e}")
    except Exception as e:
        print(f"[Summarizer] Unexpected error during summary saving: {e}")

# Optional: Add a function to create the __init__.py if needed
def ensure_utils_package():
     utils_dir = os.path.dirname(__file__)
     init_path = os.path.join(utils_dir, '__init__.py')
     if not os.path.exists(init_path):
         print(f"[Setup] Creating {init_path}...")
         with open(init_path, 'w') as f:
             pass # Create empty file

ensure_utils_package() # Call on import to ensure package structure 