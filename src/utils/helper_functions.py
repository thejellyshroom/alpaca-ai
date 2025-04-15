import gc
import torch
import numpy as np
import re
import time

def unload_component(component_obj, component_name):
    """Unload a component to free up memory.
    
    Args:
        component_obj: The current instance of the component
        component_name (str): Name of the component for logging
    """
    if component_obj:
        print(f"Unloading existing {component_name}...")
        del component_obj
        gc.collect()
        print(f"{component_name} unloaded successfully.")
        return None
    else:
        print(f"No existing {component_name} found")
        return None

def split_into_sentences(text):
    """Split text into sentences for more natural speech with pauses."""
    # Split on sentence endings (., !, ?) followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def should_use_rag(text):
    """Determine if RAG should be used based on query content.
    
    Args:
        text (str): User query text
        
    Returns:
        bool: True if RAG should be used
    """
    # Simple heuristic: use RAG for questions or when specific keywords are present
    question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'explain', 'tell me about'}
    return any(word in text.lower() for word in question_words)

def process_streaming_response(tts_handler, audio_handler, response_stream, tts_enabled=True):
    """Process a streaming response from the LLM, handling chunking and TTS.
    
    Args:
        tts_handler: The TTS handler instance
        audio_handler: The audio handler instance
        response_stream: Generator yielding response chunks
        tts_enabled (bool): Whether TTS is enabled
        
    Returns:
        str: The complete assembled response
    """
    partial_buffer = ""
    char_count = 0
    waiting_for_punctuation = False
    assistant_buffer = ""  # Store the complete response
    initial_words_done = False  # Flag to track if we've processed the first 8 words
    
    print("Assistant: ", end="", flush=True)
    
    try:
        # Get streaming response from LLM
        for token in response_stream:
            print(token, end="", flush=True)
            partial_buffer += token
            assistant_buffer += token
            char_count += len(token)
            
            # Count words when spaces are encountered
            if not initial_words_done:
                # Count words by splitting the current buffer
                words_so_far = len(partial_buffer.split())
                if words_so_far >= 8 or any(punct in token for punct in [".", "!", "?"]):
                    # Synthesize and play initial words immediately
                    if tts_enabled:
                        audio_array, sample_rate = tts_handler.synthesize(partial_buffer)
                        if audio_array is not None and len(audio_array) > 0:
                            audio_handler.play_audio(audio_array, sample_rate)
                    
                    # Reset partial buffer and mark initial words as done
                    partial_buffer = ""
                    char_count = 0
                    initial_words_done = True
                    waiting_for_punctuation = False
            # After initial words are processed
            elif waiting_for_punctuation:
                # If we see punctuation, treat that as a sentence boundary
                if any(punct in token for punct in [".", "!", "?"]):
                    # Synthesize and play this sentence
                    if tts_enabled:
                        audio_array, sample_rate = tts_handler.synthesize(partial_buffer)
                        if audio_array is not None and len(audio_array) > 0:
                            audio_handler.play_audio(audio_array, sample_rate)
                    
                    # Reset partial buffer
                    partial_buffer = ""
                    char_count = 0
                    waiting_for_punctuation = False
            # If we're not waiting for punctuation yet but initial words are done
            elif initial_words_done:
                # Once we've accumulated ~100 characters, start waiting for punctuation
                if char_count >= 100:
                    waiting_for_punctuation = True
    except Exception as e:
        print(f"\nError during LLM response generation: {e}")
        
    # Process any remaining text in the buffer
    if partial_buffer.strip() and tts_enabled:
        try:
            audio_array, sample_rate = tts_handler.synthesize(partial_buffer)
            if audio_array is not None and len(audio_array) > 0:
                audio_handler.play_audio(audio_array, sample_rate)
        except Exception as e:
            print(f"Error synthesizing final text segment: {e}")
    
    return assistant_buffer
