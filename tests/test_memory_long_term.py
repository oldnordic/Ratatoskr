"""Unit tests for the long-term memory subsystem."""

import sys
import types

# Provide dummy langchain modules so memory.long_term imports succeed
sys.modules['langchain_huggingface'] = types.SimpleNamespace(
    HuggingFaceEmbeddings=lambda model_name: None
)

class DummyFAISS:
    def __init__(self, texts=None):
        self.texts = texts or []
        self.index = types.SimpleNamespace(ntotal=len(self.texts))

    @classmethod
    def load_local(cls, *args, **kwargs):
        return cls(["Initial memory entry."])

    @classmethod
    def from_texts(cls, texts, embeddings):
        return cls(texts)

    def save_local(self, path):
        pass

    def add_texts(self, texts):
        self.texts.extend(texts)
        self.index.ntotal = len(self.texts)

    def as_retriever(self, search_kwargs=None):
        class Retriever:
            def __init__(self, texts):
                self.texts = texts

            def invoke(self, query):
                return [types.SimpleNamespace(page_content=t) for t in self.texts if query in t]

        return Retriever(self.texts)

sys.modules['langchain_community.vectorstores'] = types.SimpleNamespace(FAISS=DummyFAISS)

import memory.long_term as lt


def test_add_and_retrieve(monkeypatch):
    store = DummyFAISS([])
    monkeypatch.setattr(lt, 'get_vector_store', lambda: store)
    result = lt.add_memory('hello world')
    assert 'successfully' in result.lower()
    text = lt.retrieve_relevant_memories('hello')
    assert 'hello world' in text
