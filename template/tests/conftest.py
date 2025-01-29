"""Common test fixtures for the RAG template project."""

from typing import List
import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from unittest.mock import MagicMock


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings model."""
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return embeddings


@pytest.fixture
def mock_vector_store(mock_embeddings):
    """Create a mock vector store."""
    store = MagicMock(spec=VectorStore)
    store.embeddings = mock_embeddings

    # Sample documents for similarity search
    docs = [
        Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        Document(page_content="Test document 2", metadata={"source": "test2.txt"}),
        Document(page_content="Test document 3", metadata={"source": "test3.txt"}),
        Document(page_content="Test document 4", metadata={"source": "test4.txt"}),
    ]

    def mock_similarity_search(query: str, k: int = 4) -> List[Document]:
        return docs[:k]

    def mock_similarity_search_with_score(query: str, k: int = 4):
        return [(doc, 0.8) for doc in docs[:k]]

    store.similarity_search = mock_similarity_search
    store.similarity_search_with_score = mock_similarity_search_with_score
    return store


@pytest.fixture
def mock_artifact_store():
    """Create a mock artifact store."""
    store = MagicMock()
    store.open = MagicMock()
    return store


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="This is a test response")
    return llm
