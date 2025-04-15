import ollama
from typing import Dict, Any, Generator
import random
import sys, chromadb, ollama

class LLMHandler:
    def __init__(self, model_name='gemma3:4b', config=None):
        self.model_name = model_name
        
        # Default parameters for text generation running locally
        local_config = config.get('local', {})
        # create parameters to send to ollama
        self.params = {
            'temperature': local_config.get('temperature'),
            'top_p': local_config.get('top_p'),
            'top_k': local_config.get('top_k'),
            'max_tokens': local_config.get('max_tokens'),
            'n_ctx': local_config.get('n_ctx'),
            'repeat_penalty': local_config.get('repeat_penalty')
        }
        
        self.rag_collection = self.init_rag()
        
    def init_rag(self):
        # Initialize ChromaDB client and get collection
        chromaclient = chromadb.HttpClient(host="localhost", port=8000)
        collection = chromaclient.get_collection(name="jellyshroom")
        return collection


    def get_response(self, messages: list[Dict[str, Any]]) -> Generator[str, None, None]:
        """Get a streaming response from the LLM using conversation history.
        Yields:
            str: Chunks of the LLM's response as they are generated
        """
        # Log the generation parameters being used
        print(f"Using LLM parameters: {self.params}")
        
        # Call Ollama with our parameters
        response = ollama.chat(
            model=self.model_name, 
            messages=messages,
            stream=True,
            options=self.params
        )
        
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
                

    def get_rag_response(self, query: str, messages: list[Dict[str, Any]]) -> Generator[str, None, None]:
        """Get a streaming response from the LLM using RAG.
        
        Args:
            query (str): The user's query
            messages (list): Conversation history
        Yields:
            str: Chunks of the LLM's response as they are generated
        """
        
        # Get embedding for the query
        query_embedding = ollama.embed(model="nomic-embed-text", input=query)['embeddings']
        
        # Retrieve relevant documents
        results = self.rag_collection.query(
            query_embeddings=query_embedding,
            n_results=5  # Get top 5 most relevant documents
        )
        
        # Combine retrieved documents
        context = "\n\n".join(results['documents'][0])

        rag_prompt = f"""Context information is below.
                    ---------------------
                    {context}
                    ---------------------
                    Given the context information and not prior knowledge, answer the following question:
                    {query}
                    """

        # Add RAG prompt to messages
        rag_messages = messages.copy()
        rag_messages.append({"role": "user", "content": rag_prompt})
        
        # Get response using the enhanced context
        return self.get_response(rag_messages)