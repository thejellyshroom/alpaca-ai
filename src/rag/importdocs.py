import chromadb
from src.rag.functions import readtextfiles, chunksplitter, getembedding

def importdocs():
    collection_name="jellyshroom"

    chromaclient = chromadb.HttpClient(host="localhost", port=8000)
    textdocspath = "rag_data"
    text_data = readtextfiles(textdocspath)

    # First, check if collection exists and delete it if it does
    try:
        collection = chromaclient.get_collection(name=collection_name)
        print("Found existing collection, deleting it...")
        chromaclient.delete_collection(collection_name)
    except Exception:
        print("No existing collection found")

    # Create new collection
    print("Creating new collection...")
    collection = chromaclient.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Add documents to the collection
    for filename, text in text_data.items():
        chunks = chunksplitter(text)
        embeds = getembedding(chunks)
        chunknumber = list(range(len(chunks)))
        ids = [filename + str(index) for index in chunknumber]
        metadatas = [{"source": filename} for index in chunknumber]
        
        try:
            collection.add(ids=ids, documents=chunks, embeddings=embeds, metadatas=metadatas)
            print(f"Successfully added {len(chunks)} chunks from {filename}")
        except Exception as e:
            print(f"Error adding chunks from {filename}: {str(e)}")