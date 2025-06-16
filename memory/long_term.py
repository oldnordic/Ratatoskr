"""Long-term memory management backed by FAISS and sentence embeddings."""

import logging
import os
import threading
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Where the FAISS index is stored on disk
MEMORY_DIR = "memory"
INDEX_FILE = os.path.join(MEMORY_DIR, "ratatoskr.index")
os.makedirs(MEMORY_DIR, exist_ok=True)

# Global state used by the module.  ``vectorstore`` holds the FAISS instance and
# ``is_initialized`` guards against multiple initializations.
vectorstore = None
state_lock = threading.Lock()
is_initialized = False

try:
    temp_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    EMBEDDING_DIM = temp_model.client[0].get_sentence_embedding_dimension()
    del temp_model
except Exception:
    # Fallback dimension if sentence transformer isn't available.
    EMBEDDING_DIM = 384

def _initialize_vector_store():
    """Initializes or loads the FAISS vector store from disk."""
    global vectorstore, is_initialized
    with state_lock:
        if is_initialized:
            return
        logging.info("Initializing LangChain FAISS memory system...")
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            if os.path.exists(INDEX_FILE):
                vectorstore = FAISS.load_local(
                    MEMORY_DIR,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                vectorstore = FAISS.from_texts(["Initial memory entry."], embeddings)
                vectorstore.save_local(MEMORY_DIR)
            is_initialized = True
            logging.info("LangChain memory system initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize FAISS vector store: {e}")
            
def get_vector_store():
    """Return the global vector store, initializing it on first use."""
    if not is_initialized: _initialize_vector_store()
    return vectorstore

def retrieve_relevant_memories(query: str) -> str:
    """Retrieve stored memories that are semantically similar to ``query``."""
    vs = get_vector_store()
    if vs and vs.index.ntotal > 0:
        retriever = vs.as_retriever(search_kwargs=dict(k=2))
        results = retriever.invoke(query)
        return "\n".join([doc.page_content for doc in results])
    return "No relevant memories found."

def add_memory(text_to_store: str) -> str:
    """Save ``text_to_store`` to the vector store for later retrieval."""
    vs = get_vector_store()
    if vs:
        vs.add_texts([text_to_store])
        # Persist the updated index so the information is available across runs.
        vs.save_local(MEMORY_DIR)
        return "Information stored successfully."
    return "Failed to store information."