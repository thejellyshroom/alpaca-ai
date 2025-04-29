# src/utils/session_utils.py

from datetime import datetime
import os
import traceback
import ollama # Import ollama

from components.llm_handler import LLMHandler 
from .conversation_manager import ConversationManager 

SUMMARY_PROMPT_TEMPLATE = """
**TASK:** Objectively summarize the provided conversation history between a User and an AI Assistant.

**STYLE REQUIREMENTS:**
- Use a strictly neutral, factual, and objective tone.
- Write in the third person (e.g., "The user asked...", "The assistant explained...").
- DO NOT adopt the persona of the user or the assistant.
- DO NOT evaluate the quality of the conversation or the participants' performance.
- Focus *only* on summarizing the sequence of interactions, key topics, questions, answers, and stated goals or issues.
- Keep the summary concise and focused on the interaction flow.
- Do not add any introductory or concluding remarks like "Here is the summary:".

**Conversation History:**
{history_string}

**Concise, Objective, Third-Person Summary:**
"""

SUMMARIZER_SYSTEM_PROMPT = {"role": "system", "content": "You are an objective summarization engine. Your sole task is to create a neutral, third-person summary of the provided conversation text, focusing only on the interaction sequence and content. Adhere strictly to the formatting and style requirements provided in the user prompt. Do not inject any personality or evaluation."}

def _format_history_for_prompt(history: list[dict]) -> str:
    """Formats conversation history into a readable string."""
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

def _call_ollama_sync_for_summary(model_name: str, messages: list[dict], params: dict) -> str:
    """Calls ollama.chat synchronously and consumes the stream."""
    full_summary = ""
    try:
        response_stream = ollama.chat(
            model=model_name,
            messages=messages,
            stream=True,
            options=params
        )
        # Consume the SYNCHRONOUS stream
        for chunk in response_stream:
            if 'message' in chunk and 'content' in chunk['message']:
                full_summary += chunk['message']['content']
        return full_summary.strip()
    except Exception as e:
        print(f"[Summarizer Sync Helper] Error during Ollama call: {e}")
        traceback.print_exc()
        return "[Error generating summary via sync helper]"

def summarize_conversation(history: list[dict], llm_handler: LLMHandler) -> str:
    """Generates a summary of the conversation history using the extraction LLM synchronously."""
    if not history:
        print("[Summarizer] History is empty, skipping summarization.")
        return ""

    user_assistant_history = [msg for msg in history if msg['role'] in ('user', 'assistant')]
    if not user_assistant_history:
        print("[Summarizer] No user/assistant messages found in history, skipping summarization.")
        return ""

    print("[Summarizer] Formatting history for summarization...")
    history_string = _format_history_for_prompt(user_assistant_history)
    prompt = SUMMARY_PROMPT_TEMPLATE.format(history_string=history_string)

    summarization_messages = [
        SUMMARIZER_SYSTEM_PROMPT,
        {"role": "user", "content": prompt}
    ]

    # --- Determine model for summarization --- #
    raw_extraction_model = os.getenv('EXTRACTION_LLM_MODEL')
    extraction_model = None
    if raw_extraction_model:
        extraction_model = raw_extraction_model.split('#')[0].strip().strip('\"').strip('\'')
        model_for_summary = extraction_model
        print(f"[Summarizer] Using EXTRACTION_LLM_MODEL for summary: {model_for_summary}")
    else:
        model_for_summary = llm_handler.model_name
        print(f"[Summarizer] EXTRACTION_LLM_MODEL not set or empty. Falling back to base model: {model_for_summary}")
    # --- End Determine model --- #

    # Get params from the handler, override temperature
    summary_params = llm_handler.params.copy()
    summary_params['temperature'] = 0.3 # Override temperature for summary
    summary_params['top_p'] = 0.9

    print(f"[Summarizer] Requesting summary synchronously (model: {model_for_summary}, temp: {summary_params['temperature']})...")
    try:
        # Call the synchronous helper directly
        full_summary = _call_ollama_sync_for_summary(
            model_for_summary,
            summarization_messages,
            summary_params
        )

        print("[Summarizer] Summary received.")
        return full_summary # Already stripped in the helper

    except Exception as e:
        print(f"[Summarizer] Error during synchronous LLM call for summarization: {e}")
        traceback.print_exc()
        return "[Error generating summary]"

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

        with open(filename, "w", encoding="utf-8") as f:
            f.write(content_to_write)

    except IOError as e:
        print(f"[Summarizer] Error saving summary file {filename}: {e}")
    except Exception as e:
        print(f"[Summarizer] Unexpected error during summary saving: {e}")
