"""
Ollama LLM Interface Module
==========================

This module provides interfaces for interacting with Ollama's language models,
including text generation and embedding capabilities.

Usage:
    from llm_interfaces.ollama_interface import ollama_model_complete, ollama_embed
"""

__version__ = "1.0.0"
__author__ = "lightrag Team"
__status__ = "Production"

import sys

from minirag.base import BaseKVStorage
from minirag.utils import compute_mdhash_id

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("ollama"):
    pm.install("ollama")
if not pm.is_installed("tenacity"):
    pm.install("tenacity")

import ollama
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from minirag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
import numpy as np
from typing import Union, Optional


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def ollama_model_if_cache(
    ollama_model: str,
    prompt: str,
    system_prompt=None,
    history_messages: Optional[list] = [],
    hashing_kv: Optional[BaseKVStorage] = None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    # Let's be explicit based on the kwarg
    stream = kwargs.get("stream", True)  # Default to True if stream kwarg is missing 
                                        # (MiniRAG partial should always add stream=True now)

    # --- Caching Logic --- 
    cache_key = None
    # --- Add check for stream == False --- 
    if hashing_kv and not stream: # Only check/use cache if NOT streaming
        # Ensure history_messages is iterable for caching key
        local_history = history_messages if history_messages is not None else [] 
        cache_input = {
            "model": ollama_model,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "history": tuple(tuple(msg.items()) for msg in local_history), # Use local_history
            "kwargs": tuple(sorted(kwargs.items())) 
        }
        cache_key = compute_mdhash_id(str(cache_input), prefix="llmcache-")
        cached_response = await hashing_kv.get_by_id(cache_key)
        if cached_response:
            print(f"[Cache Hit] Returning cached response for key: {cache_key}")
            # Handle potential stream vs non-stream from cache if needed later
            # For now, assume cached is non-streamed string
            if isinstance(cached_response, dict) and 'content' in cached_response:
                return cached_response['content']
            elif isinstance(cached_response, str): # Simple string cache
                 return cached_response
            else:
                 print(f"[Cache Warning] Unexpected cache format for key {cache_key}: {type(cached_response)}")
                 # Proceed as if cache miss
    # --- End Caching Logic --- 

    # kwargs.pop("max_tokens", None)
    # kwargs.pop("response_format", None) # allow json
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    # kwargs.pop("hashing_kv", None) # Don't pop it, we might use it below
    api_key = kwargs.pop("api_key", None)
    headers = (
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        if api_key
        else {"Content-Type": "application/json"}
    )
    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Ensure history_messages is treated as a list here too, just in case
    messages.extend(history_messages if history_messages is not None else []) 
    messages.append({"role": "user", "content": prompt})

    # Remove ollama_model if it's still in kwargs 
    kwargs.pop("ollama_model", None) 

    # --- API Call --- 
    response = await ollama_client.chat(model=ollama_model, messages=messages, **kwargs)
    # --- End API Call Modification --- 
    
    # --- Caching Response --- 
    non_stream_response_content = None
    # --- Add check for stream == False --- 
    if not stream:
        non_stream_response_content = response["message"]["content"]
        # --- Only cache if NOT streaming and caching is enabled --- 
        if hashing_kv and cache_key: 
             print(f"[Cache Write] Caching response for key: {cache_key}")
             await hashing_kv.upsert({cache_key: {"content": non_stream_response_content}})
    # --- End Caching Modification ---

    if stream:
        async def inner():
            try:
                async for chunk in response:
                    if chunk.get('message') and chunk['message'].get('content'):
                        yield chunk['message']['content']
            except Exception as e:
                 print(f"\nError during Ollama stream processing: {e}")
                 yield f"[Error during streaming: {e}]"
        return inner()
    else:
        return non_stream_response_content


async def ollama_model_complete(
    user_input: str,
    ollama_model: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant.",
    hashing_kv: Optional[BaseKVStorage] = None,
    history_messages: Optional[list] = [],
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
        
    # --- Check if ollama_model was passed --- 
    if not ollama_model:
        # Attempt to get from kwargs as a fallback (though ideally it's passed directly)
        ollama_model = kwargs.pop("ollama_model", None)
        if not ollama_model:
            print("[Error] ollama_model not provided to ollama_model_complete. Falling back to 'gemma3:12b'.")
            ollama_model = 'gemma3:12b' # Fallback if not passed directly or in kwargs
    # --- End Check --- 
    
    # Pass the explicit ollama_model and the rest of the arguments
    return await ollama_model_if_cache(
        ollama_model=ollama_model,
        prompt=user_input,
        system_prompt=system_prompt,
        history_messages=history_messages,
        hashing_kv=hashing_kv,
        **kwargs,
    )


async def ollama_embedding(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    """
    Deprecated in favor of `embed`.
    """
    embed_text = []
    ollama_client = ollama.Client(**kwargs)
    for text in texts:
        data = ollama_client.embeddings(model=embed_model, prompt=text)
        embed_text.append(data["embedding"])

    return embed_text


async def ollama_embed(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    api_key = kwargs.pop("api_key", None)
    headers = (
        {"Content-Type": "application/json", "Authorization": api_key}
        if api_key
        else {"Content-Type": "application/json"}
    )
    kwargs["headers"] = headers
    ollama_client = ollama.Client(**kwargs)
    data = ollama_client.embed(model=embed_model, input=texts)
    return data["embeddings"]
