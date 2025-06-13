import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import time
import os
import json
import threading

# --- File Paths for Persistent FAISS-based Memory ---
MEMORY_DIR = "memory"
INDEX_FILE = os.path.join(MEMORY_DIR, "ratatoskr.index")
DOCS_FILE = os.path.join(MEMORY_DIR, "ratatoskr_docs.json")

# Ensure the memory directory exists
os.makedirs(MEMORY_DIR, exist_ok=True)

# --- Global State for the Memory System ---
embedding_model = None
faiss_index = None
doc_store = []
state_lock = threading.Lock()

# Determine the embedding dimension from the model
try:
    temp_model = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDING_DIM = temp_model.get_sentence_embedding_dimension()
    del temp_model
except Exception as e:
    logging.error(f"Could not determine embedding dimension, defaulting to 384. Error: {e}")
    EMBEDDING_DIM = 384

def _initialize_memory():
    """Initializes or loads the FAISS index and document store from files."""
    global faiss_index, doc_store
    with state_lock:
        if faiss_index is None:
            logging.info("Initializing FAISS memory system...")
            try:
                if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
                    logging.info(f"Loading existing memory from {INDEX_FILE} and {DOCS_FILE}...")
                    faiss_index = faiss.read_index(INDEX_FILE)
                    with open(DOCS_FILE, 'r', encoding='utf-8') as f:
                        doc_store = json.load(f)
                    logging.info(f"Successfully loaded {len(doc_store)} memories.")
                else:
                    logging.info("No existing memory found. Creating new FAISS index.")
                    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
                    doc_store = []
            except Exception as e:
                logging.error(f"Failed to initialize memory. Starting fresh. Error: {e}")
                faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
                doc_store = []

def get_embedding_model():
    """Lazily loads the SentenceTransformer model."""
    global embedding_model
    with state_lock:
        if embedding_model is None:
            logging.info("Loading sentence-transformer model (all-MiniLM-L6-v2) on demand...")
            try:
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
            except Exception:
                logging.warning("Failed to load sentence-transformer on GPU, falling back to CPU.")
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

def store_memory(text_chunk: str):
    """Stores a text chunk and its embedding."""
    _initialize_memory() # Ensure memory is loaded
    model = get_embedding_model()
    if not isinstance(faiss_index, faiss.Index): return

    logging.info(f"Storing memory: '{text_chunk[:50]}...'")
    embedding = model.encode([text_chunk]) # Encode as a list
    
    with state_lock:
        faiss_index.add(embedding.astype('float32'))
        doc_store.append(text_chunk)
        
        # Persist changes to disk
        faiss.write_index(faiss_index, INDEX_FILE)
        with open(DOCS_FILE, 'w', encoding='utf-8') as f:
            json.dump(doc_store, f)
    logging.info("Memory stored and saved to disk.")

def retrieve_relevant_memories(query_text: str, n_results: int = 2) -> list:
    """Retrieves relevant memories using FAISS."""
    _initialize_memory()
    model = get_embedding_model()
    if not isinstance(faiss_index, faiss.Index) or faiss_index.ntotal == 0:
        return []

    logging.info(f"Searching memory for query: '{query_text[:50]}...'")
    query_embedding = model.encode([query_text])
    
    # Search the FAISS index
    distances, indices = faiss_index.search(query_embedding.astype('float32'), k=min(n_results, faiss_index.ntotal))
    
    # Retrieve the corresponding documents
    relevant_docs = [doc_store[i] for i in indices[0]]
    logging.info(f"Found relevant memories: {relevant_docs}")
    return relevant_docs