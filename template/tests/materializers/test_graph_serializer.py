"""Tests for graph serialization and materialization."""

import json
import os
from typing import Dict
import pytest
from unittest.mock import MagicMock, patch
from langgraph.graph import StateGraph

from materializers.graph_serializer import GraphSerializer, SerializationError, DeserializationError
from materializers.langgraph_materializer import LangGraphMaterializer
from custom_types.state import State


@pytest.fixture
def mock_graph():
    """Create a mock graph for testing."""
    graph = StateGraph(State)
    graph.add_node("test_node", lambda x: x)
    graph.add_edge("test_node", "END")
    graph.set_entry_point("test_node")
    return graph.compile()


@pytest.fixture
def mock_artifact_store():
    """Create a mock artifact store."""
    store = MagicMock()
    store.open = MagicMock()
    return store


def test_graph_serializer_serialize(mock_graph):
    """Test graph serialization."""
    # Test basic serialization
    serialized = GraphSerializer.serialize_graph(mock_graph)
    assert isinstance(serialized, dict)
    assert "nodes" in serialized
    assert "edges" in serialized
    assert "branches" in serialized
    assert "compiled" in serialized

    # Test error handling
    with pytest.raises(SerializationError):
        GraphSerializer.serialize_graph("not a graph")


def test_graph_serializer_deserialize(mock_graph):
    """Test graph deserialization."""
    # Test basic deserialization
    serialized = GraphSerializer.serialize_graph(mock_graph)
    deserialized = GraphSerializer.deserialize_graph(serialized)
    assert isinstance(deserialized, StateGraph)
    assert hasattr(deserialized, "nodes")
    assert hasattr(deserialized, "edges")
    assert hasattr(deserialized, "branches")
    assert hasattr(deserialized, "compiled")

    # Test error handling
    with pytest.raises(DeserializationError):
        GraphSerializer.deserialize_graph({"invalid": "data"})


def test_materializer_save(mock_graph, mock_artifact_store):
    """Test LangGraphMaterializer save functionality."""
    materializer = LangGraphMaterializer(mock_artifact_store)
    
    # Mock the file operations
    mock_file = MagicMock()
    mock_artifact_store.open.return_value.__enter__.return_value = mock_file
    
    # Test saving
    materializer.save(mock_graph)
    
    # Verify the file was opened and written to
    mock_artifact_store.open.assert_called_once()
    assert mock_file.write.called


def test_materializer_load(mock_graph, mock_artifact_store):
    """Test LangGraphMaterializer load functionality."""
    materializer = LangGraphMaterializer(mock_artifact_store)
    
    # Create mock serialized data
    serialized_data = GraphSerializer.serialize_graph(mock_graph)
    
    # Mock file operations for loading
    mock_file = MagicMock()
    mock_file.read.return_value = json.dumps(serialized_data)
    mock_artifact_store.open.return_value.__enter__.return_value = mock_file
    
    # Test loading
    loaded_graph = materializer.load(StateGraph)
    
    # Verify the correct methods were called
    mock_artifact_store.open.assert_called_once()
    assert mock_file.read.called
    assert isinstance(loaded_graph, StateGraph)


def test_materializer_extract_metadata(mock_graph):
    """Test LangGraphMaterializer metadata extraction."""
    materializer = LangGraphMaterializer(MagicMock())
    
    metadata = materializer.extract_metadata(mock_graph)
    assert isinstance(metadata, dict)
    assert "num_nodes" in metadata
    assert "num_edges" in metadata
    assert "num_branches" in metadata
    assert "is_compiled" in metadata 
