"""
JsonDocStatus Storage Module
=======================

This module provides a storage interface for graphs using NetworkX, a popular Python library for creating, manipulating, and studying the structure, dynamics, and functions of complex networks.

The `NetworkXStorage` class extends the `BaseGraphStorage` class from the LightRAG library, providing methods to load, save, manipulate, and query graphs using NetworkX.

Author: lightrag team
Created: 2024-01-25
License: MIT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Version: 1.0.0

Dependencies:
    - NetworkX
    - NumPy
    - LightRAG
    - graspologic

Features:
    - Load and save graphs in various formats (e.g., GEXF, GraphML, JSON)
    - Query graph nodes and edges
    - Calculate node and edge degrees
    - Embed nodes using various algorithms (e.g., Node2Vec)
    - Remove nodes and edges from the graph

Usage:
    from minirag.storage.networkx_storage import NetworkXStorage

"""

import aiofiles
import json
import os
from dataclasses import dataclass, field
import threading

from minirag.utils import (
    logger,
    load_json,
    write_json,
)

from minirag.base import (
    BaseKVStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        self._lock = threading.Lock()
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    async def index_done_callback(self):
        with self._lock:
            data_to_write = self._data.copy()
        write_json(data_to_write, self._file_name)

    async def get_by_id(self, id):
        with self._lock:
            return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        with self._lock:
            if fields is None:
                return [self._data.get(id, None) for id in ids]
            return [
                (
                    {k: v for k, v in self._data[id].items() if k in fields}
                    if self._data.get(id, None)
                    else None
                )
                for id in ids
            ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        with self._lock:
            return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        with self._lock:
            self._data.update(data)

    async def drop(self):
        with self._lock:
            self._data = {}

    async def filter(self, filter_func):
        """Filter key-value pairs based on a filter function

        Args:
            filter_func: The filter function, which takes a value as an argument and returns a boolean value

        Returns:
            Dict: Key-value pairs that meet the condition
        """
        result = {}
        with self._lock:
            for key, value in self._data.items():
                if filter_func(value):
                    result[key] = value
        return result

    async def delete(self, ids: list[str]):
        """Delete data with specified IDs

        Args:
            ids: List of IDs to delete
        """
        with self._lock:
            for id in ids:
                if id in self._data:
                    del self._data[id]
            await self.index_done_callback()
            logger.info(f"Successfully deleted {len(ids)} items from {self.namespace}")
