"""Tests for ingest and embed step."""

import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

from config.constants import VectorStoreType
from steps.ingest_and_embed import ingest_and_embed
from utils.vector_store_factory import (
    UnsupportedStoreError,
    StoreCreationError,
)


@pytest.fixture
def mock_docs_dir(tmp_path):
    """Create a temporary directory with mock markdown files."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Create test markdown files
    (docs_dir / "test1.md").write_text("# Test Document 1\nThis is a test.")
    (docs_dir / "test2.md").write_text("# Test Document 2\nAnother test.")

    return str(docs_dir)


@pytest.fixture
def mock_store_dir(tmp_path):
    """Create a temporary directory for vector store."""
    store_dir = tmp_path / "vector_store"
    store_dir.mkdir()
    return str(store_dir)


def test_ingest_and_embed_success(mock_docs_dir, mock_store_dir):
    """Test successful document ingestion and embedding."""
    config = ingest_and_embed(
        data_path=mock_docs_dir,
        store_path=mock_store_dir,
        store_type=VectorStoreType.FAISS,
    )

    assert config.store_type == VectorStoreType.FAISS
    assert config.store_path == mock_store_dir
    assert config.documents_path == mock_docs_dir
    assert os.path.exists(mock_store_dir)


def test_ingest_and_embed_empty_dir(tmp_path):
    """Test ingestion with empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No markdown files found"):
        ingest_and_embed(
            data_path=str(empty_dir),
            store_path=str(tmp_path / "store"),
        )


def test_ingest_and_embed_invalid_store_type(mock_docs_dir, mock_store_dir):
    """Test ingestion with unsupported store type."""
    with pytest.raises(UnsupportedStoreError, match="Unsupported vector store type"):
        ingest_and_embed(
            data_path=mock_docs_dir,
            store_path=mock_store_dir,
            store_type=VectorStoreType.PINECONE,
        )


def test_ingest_and_embed_invalid_chunk_settings(mock_docs_dir, mock_store_dir):
    """Test ingestion with invalid chunk settings."""
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        ingest_and_embed(
            data_path=mock_docs_dir,
            store_path=mock_store_dir,
            chunk_size=100,
            chunk_overlap=200,
        )


def test_ingest_and_embed_nonexistent_dir():
    """Test ingestion with non-existent directory."""
    with pytest.raises(FileNotFoundError):
        ingest_and_embed(
            data_path="/nonexistent/path",
            store_path="/tmp/store",
        )
