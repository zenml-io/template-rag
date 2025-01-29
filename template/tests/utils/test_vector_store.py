"""Tests for vector store utility functions."""

from typing import Dict, Any
import pytest
from langchain_core.documents import Document

from utils.vector_store import create_retrieve, extract_store_data, find_vector_store
from config.constants import DEFAULT_SEARCH_RESULTS


def test_create_retrieve(mock_vector_store):
    """Test create_retrieve function."""
    # Create retrieval function
    retrieve_fn = create_retrieve(mock_vector_store)

    # Test with default k
    state = {"question": "test question"}
    result = retrieve_fn(state)

    assert isinstance(result, dict)
    assert "context" in result
    assert isinstance(result["context"], list)
    assert len(result["context"]) == DEFAULT_SEARCH_RESULTS
    assert all(isinstance(doc, Document) for doc in result["context"])


def test_create_retrieve_custom_k(mock_vector_store):
    """Test create_retrieve function with custom k value."""
    k = 1
    retrieve_fn = create_retrieve(mock_vector_store, k=k)

    state = {"question": "test question"}
    result = retrieve_fn(state)

    assert len(result["context"]) == k


def test_extract_store_data(mock_vector_store):
    """Test extract_store_data function."""
    texts, metadatas, embeddings = extract_store_data(mock_vector_store, k=2)

    assert isinstance(texts, list)
    assert isinstance(metadatas, list)
    assert isinstance(embeddings, list)
    assert len(texts) == len(metadatas) == len(embeddings) == 2

    # Check content types
    assert all(isinstance(text, str) for text in texts)
    assert all(isinstance(metadata, dict) for metadata in metadatas)
    assert all(isinstance(embedding, list) for embedding in embeddings)


def test_find_vector_store_direct(mock_vector_store):
    """Test find_vector_store with direct vector store object."""
    result = find_vector_store(mock_vector_store)
    assert result == mock_vector_store


def test_find_vector_store_in_dict(mock_vector_store):
    """Test find_vector_store with vector store in dictionary."""
    test_dict = {"store": mock_vector_store, "other": "value"}
    result = find_vector_store(test_dict)
    assert result == mock_vector_store


def test_find_vector_store_in_closure(mock_vector_store):
    """Test find_vector_store with vector store in closure."""

    def wrapper():
        return mock_vector_store

    result = find_vector_store(wrapper)
    assert result == mock_vector_store


def test_find_vector_store_in_object(mock_vector_store):
    """Test find_vector_store with vector store in object attributes."""

    class TestClass:
        def __init__(self):
            self.store = mock_vector_store

    obj = TestClass()
    result = find_vector_store(obj)
    assert result == mock_vector_store


def test_find_vector_store_not_found():
    """Test find_vector_store when no vector store is present."""
    test_obj = {"key": "value"}
    result = find_vector_store(test_obj)
    assert result is None


def test_find_vector_store_circular_reference(mock_vector_store):
    """Test find_vector_store with circular reference."""
    d1: Dict[str, Any] = {}
    d2: Dict[str, Any] = {"store": mock_vector_store}
    d1["ref"] = d2
    d2["ref"] = d1

    result = find_vector_store(d1)
    assert result == mock_vector_store
