import chromadb
from sentence_transformers import SentenceTransformer
import logging
import time

# Initialize the database and model once to be efficient
try:
    logging.info("Initializing ChromaDB client...")
    client = chromadb.Client() 
    collection = client.get_or_create_collection(name="mycroft_long_term_memory")
    logging.info("ChromaDB client initialized successfully.")

    logging.info("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
    logging.info("Sentence-transformer model loaded successfully.")

except Exception as e:
    logging.error(f"Failed to initialize memory module: {e}")
    client = None
    collection = None
    embedding_model = None

def store_memory(text_chunk: str, metadata: dict = None):
    """
    Stores a chunk of text in the long-term vector memory.
    """
    if not embedding_model or not collection:
        logging.error("Memory module not initialized. Cannot store memory.")
        return
    if not text_chunk.strip():
        return
        
    chunk_id = str(hash(text_chunk))
    
    logging.info(f"Creating embedding for text chunk: '{text_chunk[:30]}...'")
    embedding = embedding_model.encode(text_chunk).tolist()
    logging.info("Embedding created.")
    
    final_metadata = metadata if metadata is not None else {}
    if not final_metadata:
        final_metadata['timestamp'] = time.time()

    collection.add(
        embeddings=[embedding],
        documents=[text_chunk],
        metadatas=[final_metadata],
        ids=[chunk_id]
    )
    logging.info(f"Stored memory: '{text_chunk}' with metadata: {final_metadata}")

def retrieve_relevant_memories(query_text: str, n_results: int = 2) -> list:
    """
    Retrieves the most relevant memories from the vector store based on a query.
    """
    # This is the corrected function
    if not embedding_model or not collection:
        logging.error("Memory module not initialized. Cannot retrieve memories.")
        return []
    if not query_text.strip():
        return []
        
    logging.info(f"Creating embedding for query: '{query_text[:30]}...'")
    query_embedding = embedding_model.encode(query_text).tolist()
    logging.info("Query embedding created.")
    
    logging.info("Querying ChromaDB for relevant memories...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    logging.info("ChromaDB query complete.")
    
    return results['documents'][0] if results and results['documents'] else []