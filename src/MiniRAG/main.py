# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import os
from minirag import MiniRAG, QueryParam
from minirag.llm.hf import (
    hf_model_complete, # Keep HF complete
    hf_embed,
)
from minirag.llm.ollama import ollama_model_complete # Import ollama function
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer # Keep for embedding
from datetime import datetime

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXTRACTION_LLM_MODEL = "llama3.1"
QUERY_LLM_MODEL = "gemma3:4b"

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--outputpath", type=str, default="./src/MiniRAG/logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./src/MiniRAG/LiHua-World")
    parser.add_argument("--datapath", type=str, default="./src/MiniRAG/dataset/LiHua-World/data/")
    parser.add_argument(
        "--querypath", type=str, default="./src/MiniRAG/dataset/LiHua-World/qa/query_set.csv"
    )
    args = parser.parse_args()
    return args


args = get_args()

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
# QUERY_PATH = args.querypath # Not used in this simplified script anymore
# OUTPUT_PATH = args.outputpath # Not used in this simplified script anymore
print("USING EXTRACTION LLM (HF):", EXTRACTION_LLM_MODEL) # Changed log to HF
print("USING QUERY LLM (Ollama):", QUERY_LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# --- Instance 1: For Extraction (using Ollama Model - iodose/nuextract) ---
print(f"\n--- Initializing MiniRAG for Extraction ({EXTRACTION_LLM_MODEL}) ---")
rag_extractor = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete, # Use Ollama function
    # llm_model_func=hf_model_complete, # Use HF function
    # llm_model_name=EXTRACTION_LLM_MODEL, # Pass model name directly for HF
    # llm_model_max_token_size=200, # Check if relevant for MiniCPM
    # llm_model_max_async=1, # Adjust concurrency if needed for HF model
    # llm_model_kwargs={"device_map": "auto"}, # Try automatic device placement
    # llm_model_name=EXTRACTION_LLM_MODEL, # Not used for Ollama func
    llm_model_max_token_size=200, # Check if relevant
    llm_model_max_async=1, # Adjust concurrency if needed
    llm_model_kwargs={"ollama_model": EXTRACTION_LLM_MODEL}, # Pass model name via kwargs for Ollama
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed( # Still use HF for embeddings
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
)

# --- Indexing Phase (using rag_extractor) ---
print("\n--- Starting Indexing Phase ---")
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories like .git
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            # Skip hidden files (like .DS_Store) and macOS metadata files (._)
            if file.endswith(".txt") and not file.startswith('._') and not file.startswith('.'):
                txt_files.append(os.path.join(root, file))
            elif not file.endswith(".txt") and not file.startswith('.'): # Log skipped non-txt files
                print(f"[Info] Skipping non-txt file: {os.path.join(root, file)}")
            elif file.startswith('.'): # Log skipped hidden/metadata files
                print(f"[Info] Skipping hidden/metadata file: {os.path.join(root, file)}")
    return txt_files


WEEK_LIST = find_txt_files(DATA_PATH)
print(f"Found {len(WEEK_LIST)} files to process: {WEEK_LIST}")
for i, WEEK in enumerate(WEEK_LIST):
    id = WEEK_LIST.index(WEEK) # Or simply use i
    print(f"--- Processing file {i}/{len(WEEK_LIST)}: {WEEK} ---")
    try:
        print(f"[{datetime.now()}] Reading file... {WEEK}")
        with open(WEEK) as f: # Consider adding encoding='latin-1' if needed again
            content = f.read()
        print(f"[{datetime.now()}] Starting rag_extractor.insert for file {i}...")
        rag_extractor.insert(content) # Use the extractor instance
        print(f"[{datetime.now()}] Finished rag_extractor.insert for file {i}.")
    except Exception as e:
        print(f"Error processing file {WEEK}: {e}")
        # Optionally continue to the next file or break
        # continue
        break # Stop on error for now

print("\n--- Indexing Phase Complete ---")

# Cleanup extractor instance (optional, helps release memory if large models were loaded)
del rag_extractor 

# Ensure working directory exists before initializing querier (Fix for FileNotFoundError)
os.makedirs(WORKING_DIR, exist_ok=True)

# --- Instance 2: For Querying (using Ollama Model) ---
print(f"\n--- Initializing MiniRAG for Querying ({QUERY_LLM_MODEL}) ---")
rag_querier = MiniRAG(
    working_dir=WORKING_DIR, # Use the SAME working directory
    llm_model_func=ollama_model_complete, # Use ollama function
    llm_model_max_token_size=200, # May not be relevant for Ollama
    llm_model_max_async=1, # Keep concurrency at 1
    # llm_model_name=QUERY_LLM_MODEL, # Not needed here
    llm_model_kwargs={"ollama_model": QUERY_LLM_MODEL}, # Pass ollama model name
    # Embedding func is needed for query embedding, even if index exists
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
    # We assume the index/graph already exists, no need to re-insert
)

# --- Querying Phase (using rag_querier) ---
print("\n--- Starting Query Phase ---")
# A toy query
query = "Who did Lihua get lunch with?" # Test query
answer = (
    rag_querier.query(query, param=QueryParam(mode="mini")).replace("\n", "").replace("\r", "") # Use the querier instance
)
print(f"Query: {query}")
print(f"Answer: {answer}")

print("\n--- Script Finished ---")
