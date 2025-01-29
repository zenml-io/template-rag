"""Tests for vector store utility functions."""

from typing import List

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.constants import MAX_SEARCH_RESULTS
from utils.vector_store import create_retrieve, extract_store_data, find_vector_store


class MockEmbeddings(Embeddings):
    """Mock embeddings for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Mock embedding function."""
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Mock query embedding function."""
        return [0.1, 0.2, 0.3]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    texts = ["test document 1", "test document 2"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]
    embeddings = MockEmbeddings()
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)


def test_create_retrieve(mock_vector_store):
    """Test create_retrieve function."""
    # Create retrieval function
    retrieve = create_retrieve(mock_vector_store)

    # Test retrieval
    state = {"question": "test"}
    result = retrieve(state)

    assert "context" in result
    assert isinstance(result["context"], list)
    assert len(result["context"]) <= MAX_SEARCH_RESULTS
    assert all(isinstance(doc, Document) for doc in result["context"])


def test_extract_store_data(mock_vector_store):
    """Test extract_store_data function."""
    # Extract data
    texts, metadatas, embeddings = extract_store_data(mock_vector_store, k=2)

    # Check results
    assert len(texts) == 2
    assert len(metadatas) == 2
    assert len(embeddings) == 2
    assert all(isinstance(text, str) for text in texts)
    assert all(isinstance(meta, dict) for meta in metadatas)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(isinstance(val, float) for emb in embeddings for val in emb)


def test_find_vector_store(mock_vector_store):
    """Test find_vector_store function."""
    # Test direct vector store
    found = find_vector_store(mock_vector_store)
    assert found == mock_vector_store

    # Test nested in dict
    nested_dict = {"store": mock_vector_store}
    found = find_vector_store(nested_dict)
    assert found == mock_vector_store

    # Test nested in list
    nested_list = [1, "test", mock_vector_store]
    found = find_vector_store(nested_list)
    assert found == mock_vector_store

    # Test nested in object
    class TestObj:
        def __init__(self):
            self.store = mock_vector_store

    obj = TestObj()
    found = find_vector_store(obj)
    assert found == mock_vector_store

    # Test not found
    not_found = find_vector_store({"test": "value"})
    assert not_found is None

    # Test circular reference
    circular = {"self": None}
    circular["self"] = circular
    found = find_vector_store(circular)
    assert found is None
