"""Tests for vector store factory."""

import pytest
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document

from config.constants import VectorStoreType
from config.store import VectorStoreConfig
from utils.vector_store_factory import (
    VectorStoreFactory,
    UnsupportedStoreError,
    StoreCreationError,
)
from tests.utils.test_vector_store import MockEmbeddings


@pytest.fixture
def mock_config():
    """Create a mock vector store config."""
    return VectorStoreConfig(
        store_type=VectorStoreType.FAISS,
        store_path="test/path",
        embedding_model="text-embedding-3-small",
    )


@pytest.fixture
def mock_documents():
    """Create mock documents for testing."""
    return [
        Document(page_content="test document 1", metadata={"source": "test1"}),
        Document(page_content="test document 2", metadata={"source": "test2"}),
    ]


def test_get_store_class():
    """Test getting vector store class."""
    # Test valid store types
    assert VectorStoreFactory.get_store_class(VectorStoreType.FAISS) == FAISS
    assert VectorStoreFactory.get_store_class(VectorStoreType.CHROMA) == Chroma

    # Test invalid store type
    with pytest.raises(UnsupportedStoreError, match="Unsupported vector store type"):
        VectorStoreFactory.get_store_class(VectorStoreType.PINECONE)


def test_create_store(mock_config, mock_documents):
    """Test creating a vector store."""
    embeddings = MockEmbeddings()
    store = VectorStoreFactory.create_store(
        documents=mock_documents,
        embeddings=embeddings,
        config=mock_config,
    )
    assert isinstance(store, FAISS)

    # Test with metadata
    mock_config.metadata = {"metric": "cosine"}
    store = VectorStoreFactory.create_store(
        documents=mock_documents,
        embeddings=embeddings,
        config=mock_config,
    )
    assert isinstance(store, FAISS)


def test_load_store(mock_config, tmp_path):
    """Test loading a vector store."""
    embeddings = MockEmbeddings()

    # Test loading non-existent store
    mock_config.store_path = str(tmp_path / "nonexistent")
    store = VectorStoreFactory.load_store(mock_config, embeddings)
    assert store is None

    # Create and save a store
    mock_config.store_path = str(tmp_path / "test_store")
    store = VectorStoreFactory.create_store(
        documents=[Document(page_content="test")],
        embeddings=embeddings,
        config=mock_config,
    )
    store.save_local(mock_config.store_path)

    # Test loading existing store
    loaded_store = VectorStoreFactory.load_store(mock_config, embeddings)
    assert isinstance(loaded_store, FAISS)


def test_create_store_with_invalid_config():
    """Test creating a store with invalid config."""
    embeddings = MockEmbeddings()
    invalid_config = VectorStoreConfig(
        store_type=VectorStoreType.PINECONE,  # Unsupported type
        store_path="test/path",
        embedding_model="text-embedding-3-small",
    )

    with pytest.raises(UnsupportedStoreError, match="Unsupported vector store type"):
        VectorStoreFactory.create_store(
            documents=[Document(page_content="test")],
            embeddings=embeddings,
            config=invalid_config,
        )
