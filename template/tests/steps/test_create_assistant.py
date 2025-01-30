"""Tests for the create_assistant step."""

import pytest
from unittest.mock import MagicMock, patch
from langgraph.graph import StateGraph

from config.constants import VectorStoreType
from config.store import VectorStoreConfig, AssistantConfig
from steps.create_assistant import create_assistant
from custom_types.state import State


@pytest.fixture
def mock_vector_store_config():
    """Create a mock vector store config."""
    return VectorStoreConfig(
        store_type=VectorStoreType.FAISS,
        store_path="data/test",
        embedding_model="text-embedding-3-small",
    )


def test_create_assistant(mock_vector_store_config, mock_vector_store):
    """Test creating an assistant with default parameters."""
    with patch("steps.create_assistant.VectorStoreFactory.load_store", return_value=mock_vector_store):
        config, graph = create_assistant(vector_store_config=mock_vector_store_config)
        
        # Test config
        assert isinstance(config, AssistantConfig)
        assert config.vector_store == mock_vector_store_config
        assert config.llm.model == "gpt-3.5-turbo"
        assert config.llm.temperature == 0.7
        assert "qa" in config.prompt_templates
        
        # Test graph
        assert isinstance(graph, StateGraph)
        assert hasattr(graph, "nodes")
        assert hasattr(graph, "edges")
        assert hasattr(graph, "branches")
        assert graph.compiled
        
        # Test graph structure
        assert "retrieve" in graph.nodes
        assert "generate" in graph.nodes


def test_create_assistant_custom_params(mock_vector_store_config, mock_vector_store):
    """Test creating an assistant with custom parameters."""
    with patch("steps.create_assistant.VectorStoreFactory.load_store", return_value=mock_vector_store):
        config, graph = create_assistant(
            vector_store_config=mock_vector_store_config,
            model="gpt-4",
            temperature=0.5,
            qa_template="Custom template {context} {question}",
        )
        
        # Test custom config values
        assert config.llm.model == "gpt-4"
        assert config.llm.temperature == 0.5
        assert config.prompt_templates["qa"] == "Custom template {context} {question}"
        
        # Graph should still be properly constructed
        assert isinstance(graph, StateGraph)
        assert graph.compiled


def test_create_assistant_error_handling(mock_vector_store_config):
    """Test error handling in create_assistant."""
    # Test vector store loading failure
    with patch("steps.create_assistant.VectorStoreFactory.load_store", return_value=None):
        with pytest.raises(ValueError, match="Failed to load vector store"):
            create_assistant(vector_store_config=mock_vector_store_config)
    
    # Test invalid model
    with patch("steps.create_assistant.VectorStoreFactory.load_store", return_value=MagicMock()):
        with pytest.raises(ValueError):
            create_assistant(
                vector_store_config=mock_vector_store_config,
                model="invalid-model",
            ) 
