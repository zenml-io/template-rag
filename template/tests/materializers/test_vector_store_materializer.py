"""Tests for the InMemoryVectorStore materializer."""

import os
from typing import List
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from materializers.vector_store_materializer import InMemoryVectorStoreMaterializer
from config.constants import MAX_SEARCH_RESULTS


@pytest.fixture
def sample_texts() -> List[str]:
    """Sample texts for testing."""
    return [
        "This is the first test document",
        "This is the second test document",
        "This is the third test document",
    ]


@pytest.fixture
def sample_metadatas() -> List[dict]:
    """Sample metadata for testing."""
    return [
        {"source": "test1.txt"},
        {"source": "test2.txt"},
        {"source": "test3.txt"},
    ]


@pytest.fixture
def sample_embeddings() -> List[List[float]]:
    """Sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]


@pytest.fixture
def vector_store(
    sample_texts, sample_metadatas, sample_embeddings, mock_embeddings
) -> InMemoryVectorStore:
    """Create a sample vector store."""
    store = InMemoryVectorStore(embedding=mock_embeddings)
    store.add_texts(
        texts=sample_texts,
        metadatas=sample_metadatas,
        embeddings=sample_embeddings,
    )
    return store


def test_init_materializer(mock_artifact_store):
    """Test materializer initialization."""
    uri = "/test/path"
    materializer = InMemoryVectorStoreMaterializer(uri, mock_artifact_store)

    assert materializer.uri == uri
    assert materializer.artifact_store == mock_artifact_store
    assert materializer.store_path == os.path.join(uri, "documents.dill")


def test_save_vector_store(vector_store, mock_artifact_store):
    """Test saving a vector store."""
    uri = "/test/path"
    materializer = InMemoryVectorStoreMaterializer(uri, mock_artifact_store)

    # Create a mock file object
    mock_file = MagicMock()
    mock_artifact_store.open.return_value.__enter__.return_value = mock_file

    # Save the vector store
    materializer.save(vector_store)

    # Verify the file was opened correctly
    mock_artifact_store.open.assert_called_once_with(
        os.path.join(uri, "documents.dill"), "wb"
    )

    # Verify something was written to the file
    assert mock_file.write.called


def test_load_vector_store(
    sample_texts,
    sample_metadatas,
    sample_embeddings,
    mock_artifact_store,
    mock_embeddings,
):
    """Test loading a vector store."""
    uri = "/test/path"
    materializer = InMemoryVectorStoreMaterializer(uri, mock_artifact_store)

    # Create test data
    test_data = {
        "texts": sample_texts,
        "metadatas": sample_metadatas,
        "embeddings": sample_embeddings,
    }

    # Mock the file reading
    with patch("dill.load") as mock_load:
        mock_load.return_value = test_data
        loaded_store = materializer.load(InMemoryVectorStore)

    # Verify the loaded store
    assert isinstance(loaded_store, InMemoryVectorStore)

    # Test similarity search to verify the content
    results = loaded_store.similarity_search("test", k=3)
    assert len(results) == 3
    assert all(isinstance(doc, Document) for doc in results)


def test_save_load_roundtrip(vector_store, mock_artifact_store):
    """Test saving and loading a vector store (roundtrip test)."""
    uri = "/test/path"
    materializer = InMemoryVectorStoreMaterializer(uri, mock_artifact_store)

    # Save the vector store
    with patch("dill.dump") as mock_dump:
        materializer.save(vector_store)
        saved_data = mock_dump.call_args[0][0]

    # Load the vector store
    with patch("dill.load") as mock_load:
        mock_load.return_value = saved_data
        loaded_store = materializer.load(InMemoryVectorStore)

    # Verify the loaded store has the same content
    original_docs = vector_store.similarity_search("", k=MAX_SEARCH_RESULTS)
    loaded_docs = loaded_store.similarity_search("", k=MAX_SEARCH_RESULTS)

    assert len(original_docs) == len(loaded_docs)

    # Sort documents by content before comparing
    original_docs = sorted(original_docs, key=lambda x: x.page_content)
    loaded_docs = sorted(loaded_docs, key=lambda x: x.page_content)

    for orig_doc, loaded_doc in zip(original_docs, loaded_docs):
        assert orig_doc.page_content == loaded_doc.page_content
        assert orig_doc.metadata == loaded_doc.metadata


def test_error_handling(mock_artifact_store):
    """Test error handling in the materializer."""
    uri = "/test/path"
    materializer = InMemoryVectorStoreMaterializer(uri, mock_artifact_store)

    # Test file not found error
    mock_artifact_store.open.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        materializer.load(InMemoryVectorStore)

    # Test permission error
    mock_artifact_store.open.side_effect = PermissionError
    with pytest.raises(PermissionError):
        materializer.load(InMemoryVectorStore)
