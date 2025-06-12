import chromadb
from sentence_transformers import SentenceTransformer
import logging
import time
import os
import threading

# Use an absolute path for the database to ensure it's always found.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(project_root, "chroma_db")

logging.info(f"Using persistent database at: {db_path}")
if not os.path.exists(db_path):
    os.makedirs(db_path)

# --- CHANGE: Defer model loading ---
# Initialize the model as None. It will be loaded by the first function that needs it.
embedding_model = None
model_lock = threading.Lock() # Ensures the model is only loaded once.

def get_embedding_model():
    """Lazily loads the SentenceTransformer model on the first call."""
    global embedding_model
    with model_lock:
        if embedding_model is None:
            logging.info("Loading sentence-transformer model (all-MiniLM-L6-v2) on demand...")
            try:
                # Attempt to use the GPU
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
                logging.info("Sentence-transformer model loaded successfully onto GPU.")
            except Exception:
                logging.warning("Failed to load sentence-transformer model on GPU, falling back to CPU.")
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logging.info("Sentence-transformer model loaded successfully on CPU.")
    return embedding_model

# Initialize the persistent client
try:
    client = chromadb.PersistentClient(path=db_path) 
    collection = client.get_or_create_collection(name="ratatoskr_long_term_memory")
except Exception as e:
    logging.error(f"Failed to initialize ChromaDB client: {e}")
    client = None
    collection = None


def store_memory(text_chunk: str, metadata: dict = None):
    """Stores a chunk of text in the long-term vector memory."""
    model = get_embedding_model()
    if not model or not collection:
        return
    if not text_chunk.strip():
        return
    chunk_id = str(hash(text_chunk))
    embedding = model.encode(text_chunk).tolist()
    final_metadata = metadata if metadata is not None else {}
    if 'timestamp' not in final_metadata:
        final_metadata['timestamp'] = time.time()
    collection.add(embeddings=[embedding], documents=[text_chunk], metadatas=[final_metadata], ids=[chunk_id])
    logging.info(f"Stored memory: '{text_chunk}'")


def retrieve_relevant_memories(query_text: str, n_results: int = 2) -> list:
    """Retrieves the most relevant memories from the vector store."""
    model = get_embedding_model()
    if not model or not collection:
        return []
    if not query_text.strip():
        return []
    query_embedding = model.encode(query_text).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results['documents'][0] if results and results['documents'] else []