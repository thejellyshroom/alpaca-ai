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
