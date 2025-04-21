"""
Ollama LLM Interface Module
==========================

This module provides interfaces for interacting with Ollama's language models,
including text generation and embedding capabilities.

Author: Lightrag team
Created: 2024-01-24
License: MIT License

Copyright (c) 2024 Lightrag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

Version: 1.0.0

Change Log:
- 1.0.0 (2024-01-24): Initial release
    * Added async chat completion support
    * Added embedding generation
    * Added stream response capability

Dependencies:
    - ollama
    - numpy
    - pipmaster
    - Python >= 3.10

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
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    hashing_kv: Optional[BaseKVStorage] = None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    # --- Caching Logic --- 
    cache_key = None
    if hashing_kv:
        # Create a hashable representation of the request for caching
        cache_input = {
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "history": tuple(tuple(msg.items()) for msg in history_messages) if history_messages else None,
            "kwargs": tuple(sorted(kwargs.items())) # Ensure kwargs order doesn't break cache
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

    stream = True if kwargs.get("stream") else False
    kwargs.pop("max_tokens", None)
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
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Remove ollama_model if it's still in kwargs (shouldn't be if logic above is right, but safe)
    kwargs.pop("ollama_model", None) 

    response = await ollama_client.chat(model=model, messages=messages, **kwargs)
    
    # --- Caching Response --- 
    non_stream_response_content = None
    if not stream:
        non_stream_response_content = response["message"]["content"]
        if hashing_kv and cache_key: 
             # Cache the non-streamed response
             print(f"[Cache Write] Caching response for key: {cache_key}")
             await hashing_kv.upsert({cache_key: {"content": non_stream_response_content}})
    # --- End Caching Response --- 

    if stream:
        """cannot cache stream response yet"""
        async def inner():
            async for chunk in response:
                yield chunk["message"]["content"]
        return inner()
    else:
        return non_stream_response_content


async def ollama_model_complete(
    user_input: str,
    system_prompt: str = "You are a helpful assistant.",
    hashing_kv: Optional[BaseKVStorage] = None,
    history_messages: Optional[list] = None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
        
    # --- Corrected Model Name Logic --- 
    # Prioritize model from kwargs if available (e.g., passed via llm_model_kwargs)
    # Otherwise, use the default model from the global config (via hashing_kv)
    if hashing_kv and hasattr(hashing_kv, 'global_config') and 'llm_model_name' in hashing_kv.global_config:
        default_model_name = hashing_kv.global_config['llm_model_name']
    else:
        # Fallback if hashing_kv or its config is missing (should ideally not happen)
        default_model_name = 'gemma3:4b' # Or some other sensible default
        print(f"[Warning] Could not get default LLM model from hashing_kv.global_config. Using '{default_model_name}'.")
        
    # Get model name: Use 'ollama_model' from kwargs if present, else use the default determined above.
    model_name = kwargs.get("ollama_model", default_model_name)
    # --- End Corrected Logic ---
    
    # Pass the determined model_name and the rest of the arguments
    return await ollama_model_if_cache(
        model_name,
        user_input,
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
