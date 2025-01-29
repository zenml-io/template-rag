"""Tests for the StateGraph materializer."""

import os
from typing import List
import pytest
from unittest.mock import MagicMock, patch
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from materializers.state_graph_materializer import StateGraphMaterializer
from custom_types.state import State
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
def mock_faiss_store(mock_embeddings):
    """Create a mock FAISS store."""
    with patch("langchain_community.vectorstores.FAISS") as mock_faiss:
        store = MagicMock(spec=FAISS)
        store.embeddings = mock_embeddings
        store.similarity_search.return_value = [
            MagicMock(
                page_content=f"Test content {i}", metadata={"source": f"test{i}.txt"}
            )
            for i in range(3)
        ]
        mock_faiss.from_texts.return_value = store
        yield store


@pytest.fixture
def sample_graph(mock_faiss_store) -> CompiledStateGraph:
    """Create a sample state graph."""
    # Create a simple graph
    graph = (
        StateGraph(State)
        .add_node(
            "retrieve", lambda x: {"context": mock_faiss_store.similarity_search("")}
        )
        .add_node("generate", lambda x: {"answer": "Test response"})
        .add_edge(START, "retrieve")
        .add_edge("retrieve", "generate")
        .compile()
    )
    return graph


def test_init_materializer(mock_artifact_store):
    """Test materializer initialization."""
    uri = "/test/path"
    materializer = StateGraphMaterializer(uri, mock_artifact_store)

    assert materializer.uri == uri
    assert materializer.artifact_store == mock_artifact_store
    assert materializer.store_path == os.path.join(uri, "state_graph.dill")


def test_save_state_graph(sample_graph, mock_artifact_store):
    """Test saving a state graph."""
    uri = "/test/path"
    materializer = StateGraphMaterializer(uri, mock_artifact_store)

    # Create a mock file object
    mock_file = MagicMock()
    mock_artifact_store.open.return_value.__enter__.return_value = mock_file

    # Save the graph
    materializer.save(sample_graph)

    # Verify the file was opened correctly
    mock_artifact_store.open.assert_called_once_with(
        os.path.join(uri, "state_graph.dill"), "wb"
    )

    # Verify something was written to the file
    assert mock_file.write.called


def test_load_state_graph(
    sample_texts, sample_metadatas, mock_artifact_store, mock_embeddings
):
    """Test loading a state graph."""
    uri = "/test/path"
    materializer = StateGraphMaterializer(uri, mock_artifact_store)

    # Create test data
    test_data = {
        "texts": sample_texts,
        "metadatas": sample_metadatas,
    }

    # Mock the file reading
    with patch("dill.load") as mock_load:
        mock_load.return_value = test_data
        loaded_graph = materializer.load(CompiledStateGraph)

    # Verify the loaded graph
    assert isinstance(loaded_graph, CompiledStateGraph)


def test_save_load_roundtrip(sample_graph, mock_artifact_store):
    """Test saving and loading a state graph (roundtrip test)."""
    uri = "/test/path"
    materializer = StateGraphMaterializer(uri, mock_artifact_store)

    # Save the graph
    with patch("dill.dump") as mock_dump:
        materializer.save(sample_graph)
        saved_data = mock_dump.call_args[0][0]

    # Load the graph
    with patch("dill.load") as mock_load:
        mock_load.return_value = saved_data
        loaded_graph = materializer.load(CompiledStateGraph)

    # Test the loaded graph with a sample input
    test_input = {"question": "What is the test about?"}
    result = loaded_graph.invoke(test_input)

    assert isinstance(result, dict)
    assert "answer" in result


def test_error_handling(mock_artifact_store):
    """Test error handling in the materializer."""
    uri = "/test/path"
    materializer = StateGraphMaterializer(uri, mock_artifact_store)

    # Test file not found error
    mock_artifact_store.open.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        materializer.load(CompiledStateGraph)

    # Test permission error
    mock_artifact_store.open.side_effect = PermissionError
    with pytest.raises(PermissionError):
        materializer.load(CompiledStateGraph)


def test_vector_store_not_found(mock_artifact_store):
    """Test handling when vector store cannot be found in graph."""
    uri = "/test/path"
    materializer = StateGraphMaterializer(uri, mock_artifact_store)

    # Create a graph without a vector store
    graph = (
        StateGraph(State)
        .add_node("test", lambda x: x)
        .add_edge(START, "test")
        .compile()
    )

    # Verify that saving raises an error
    with pytest.raises(ValueError, match="Could not extract vector store from graph"):
        materializer.save(graph)
