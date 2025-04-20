# src/rag/indexer.py
import json
import os
import sys
import gc
import traceback
from minirag import MiniRAG
from minirag.llm.ollama import ollama_model_complete
from minirag.llm.hf import hf_embed
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from datetime import datetime
from dotenv import load_dotenv
import asyncio # Ensure asyncio is imported

def setup_embedding_func(model_name):
    """Initializes the embedding function using Hugging Face transformers."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        embed_model = AutoModel.from_pretrained(model_name)
        # Use a fixed embedding dimension common for MiniLM models
        # You might need to adjust if using a different embedding model type
        embedding_dim = 384 
        print(f"Embedding model '{model_name}' loaded. Dimension: {embedding_dim}")
        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=1000, # Adjust if needed
            func=lambda texts: hf_embed(texts, tokenizer=tokenizer, embed_model=embed_model),
        )
    except Exception as e:
        print(f"Error loading embedding model '{model_name}': {e}. Exiting.")
        sys.exit(1)

def find_txt_files(root_path):
    """Recursively finds all .txt files in a directory, skipping hidden files/dirs."""
    txt_files = []
    if not os.path.isdir(root_path):
        print(f"Error: Data path '{root_path}' is not a valid directory.")
        return txt_files
        
    print(f"Searching for .txt files in: {root_path}")
    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            # Skip hidden files and macOS metadata files
            if file.endswith(".txt") and not file.startswith('.') and not file.startswith('._'):
                full_path = os.path.join(root, file)
                txt_files.append(full_path)
                
    print(f"Found {len(txt_files)} .txt files to potentially process.")
    return txt_files

# Make run_indexing async
async def run_indexing():
    """Main async function to run the indexing process."""
    print("--- Starting MiniRAG Indexing Process ---")
    load_dotenv() # Load environment variables from .env

    # --- Configuration from Environment Variables ---
    WORKING_DIR = os.getenv('WORKING_DIR')
    DATA_PATH = os.getenv('DATA_PATH')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
    EXTRACTION_LLM_MODEL = os.getenv('EXTRACTION_LLM_MODEL')
    LLM_MAX_TOKEN_SIZE = int(os.getenv('LLM_MAX_TOKEN_SIZE', '200')) # Default 200
    LLM_MAX_ASYNC = int(os.getenv('LLM_MAX_ASYNC', '1')) # Default 1

    # Validate required variables
    required_vars = {'WORKING_DIR': WORKING_DIR, 'DATA_PATH': DATA_PATH, 
                     'EMBEDDING_MODEL': EMBEDDING_MODEL, 'EXTRACTION_LLM_MODEL': EXTRACTION_LLM_MODEL}
    for name, value in required_vars.items():
        if not value:
            print(f"Error: Environment variable '{name}' is not set. Please define it in .env. Exiting.")
            sys.exit(1)
            
    # Clean the extraction model name (remove comments, quotes, whitespace)
    if EXTRACTION_LLM_MODEL:
        raw_extraction_model = EXTRACTION_LLM_MODEL
        EXTRACTION_LLM_MODEL = raw_extraction_model.split('#')[0].strip().strip('"').strip("'")
    print(f"EXTRACTION_LLM_MODEL: '{EXTRACTION_LLM_MODEL}'")
    print(f"WORKING_DIR: {WORKING_DIR}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"EXTRACTION_LLM_MODEL: {EXTRACTION_LLM_MODEL}")
    print(f"LLM_MAX_TOKEN_SIZE: {LLM_MAX_TOKEN_SIZE}")
    print(f"LLM_MAX_ASYNC: {LLM_MAX_ASYNC}")
    
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # --- Initialize Embedding Function ---
    embedding_func = setup_embedding_func(EMBEDDING_MODEL)

    # --- Initialize MiniRAG for Extraction ---
    print(f"\n--- Initializing MiniRAG for Extraction ({EXTRACTION_LLM_MODEL}) ---")
    try:
        rag_extractor = MiniRAG(
            working_dir=WORKING_DIR,
            llm_model_func=ollama_model_complete, # Using Ollama for extraction
            llm_model_max_token_size=LLM_MAX_TOKEN_SIZE,
            llm_model_max_async=LLM_MAX_ASYNC,
            llm_model_kwargs={"ollama_model": EXTRACTION_LLM_MODEL}, # Pass Ollama model name
            embedding_func=embedding_func,
        )
        print("MiniRAG Extractor initialized.")
    except Exception as e:
        print(f"Error initializing MiniRAG Extractor: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # --- Load Document Processing Status ---
    kv_store_path = os.path.join(WORKING_DIR, "doc_status.json")
    doc_status = {}
    if os.path.exists(kv_store_path):
        try:
            with open(kv_store_path, "r", encoding="utf-8") as kv_file:
                doc_status = json.load(kv_file)
            print(f"Loaded processing status for {len(doc_status)} files from {kv_store_path}")
        except json.JSONDecodeError:
             print(f"Warning: Could not decode JSON from {kv_store_path}. Starting with empty status.")
        except Exception as e:
             print(f"Warning: Could not read status file {kv_store_path}: {e}. Starting with empty status.")

    # --- Indexing Phase ---
    print("\n--- Starting Indexing Phase ---")
    files_to_process = find_txt_files(DATA_PATH)
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for i, file_path in enumerate(files_to_process):
        file_key = file_path 
        if doc_status.get(file_key, {}).get("status") == "processed":
            skipped_count += 1
            continue

        print(f"--- Processing file {i+1}/{len(files_to_process)}: {file_path} --- ")
        try:
            print(f"[{datetime.now()}] Reading file content...")
            # Consider using aiofiles for async file reading if files are large
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content.strip():
                 print(f"[{datetime.now()}] Skipping empty file.")
                 doc_status[file_key] = {"status": "skipped_empty", "timestamp": str(datetime.now())}
                 skipped_count += 1
            else:
                print(f"[{datetime.now()}] Starting MiniRAG ainsert...")
                # --- Use await ainsert --- 
                await rag_extractor.ainsert(content) 
                # -------------------------
                print(f"[{datetime.now()}] Finished MiniRAG ainsert.")
                doc_status[file_key] = {"status": "processed", "timestamp": str(datetime.now())}
                processed_count += 1

            # Update status file after each file
            try:
                # Consider async file write if needed
                with open(kv_store_path, "w", encoding="utf-8") as kv_file:
                    json.dump(doc_status, kv_file, indent=4)
            except Exception as e:
                 print(f"Warning: Could not write status update to {kv_store_path}: {e}")

        except Exception as e:
            print(f"***** Error processing file {file_path}: {e} *****")
            # Log original traceback for the specific file error
            traceback.print_exc() 
            doc_status[file_key] = {"status": "error", "error_message": str(e), "timestamp": str(datetime.now())}
            error_count += 1
            # Save status even on error
            try:
                with open(kv_store_path, "w", encoding="utf-8") as kv_file:
                    json.dump(doc_status, kv_file, indent=4)
            except Exception as e_save:
                 print(f"Warning: Could not write error status update to {kv_store_path}: {e_save}")
            # Decide whether to continue or stop on error
            # break 

    print("\n--- Indexing Phase Complete ---")
    print(f"Summary: Processed={processed_count}, Skipped={skipped_count}, Errors={error_count}")

    # Optional cleanup
    del rag_extractor
    gc.collect()
    print("Indexer instance cleaned up.")