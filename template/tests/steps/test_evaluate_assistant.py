"""Tests for the evaluate_assistant step."""

import pytest
from unittest.mock import MagicMock, patch
from langgraph.graph import StateGraph

from config.store import AssistantConfig, LLMConfig, VectorStoreConfig
from steps.evaluate_assistant import evaluate_assistant, DEFAULT_TEST_QUESTIONS
from custom_types.state import State


@pytest.fixture
def mock_assistant_config():
    """Create a mock assistant config."""
    return AssistantConfig(
        vector_store=VectorStoreConfig(
            store_type="FAISS",
            store_path="data/test",
            embedding_model="text-embedding-3-small",
        ),
        llm=LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
        ),
    )


@pytest.fixture
def mock_graph():
    """Create a mock graph that returns predefined responses."""
    graph = MagicMock(spec=StateGraph)
    
    def mock_invoke(state):
        # Return different responses based on the question
        if "service connectors" in state["question"].lower():
            return {"answer": "Service connectors are..."}
        elif "deploy" in state["question"].lower():
            return {"answer": "To deploy ZenML..."}
        else:
            return {"answer": "I don't know."}
    
    graph.invoke = mock_invoke
    return graph


def test_evaluate_assistant_default_questions(mock_assistant_config, mock_graph):
    """Test evaluation with default test questions."""
    metrics = evaluate_assistant(
        config=mock_assistant_config,
        graph=mock_graph,
    )
    
    # Check metrics structure
    assert isinstance(metrics, dict)
    assert "response_count" in metrics
    assert "avg_response_length" in metrics
    assert "success_rate" in metrics
    
    # All default questions should get responses
    assert metrics["response_count"] == len(DEFAULT_TEST_QUESTIONS)
    assert metrics["success_rate"] == 1.0
    assert metrics["avg_response_length"] > 0


def test_evaluate_assistant_custom_questions(mock_assistant_config, mock_graph):
    """Test evaluation with custom test questions."""
    custom_questions = [
        "What is the meaning of life?",
        "How does this work?",
    ]
    
    metrics = evaluate_assistant(
        config=mock_assistant_config,
        graph=mock_graph,
        test_questions=custom_questions,
    )
    
    # Check metrics with custom questions
    assert metrics["response_count"] == len(custom_questions)
    assert metrics["success_rate"] == 1.0  # Mock always returns some answer


def test_evaluate_assistant_error_handling(mock_assistant_config, mock_graph):
    """Test error handling during evaluation."""
    # Mock graph to raise an exception
    error_graph = MagicMock(spec=StateGraph)
    error_graph.invoke.side_effect = Exception("Test error")
    
    metrics = evaluate_assistant(
        config=mock_assistant_config,
        graph=error_graph,
        test_questions=["Test question"],
    )
    
    # Should handle errors gracefully
    assert metrics["response_count"] == 0
    assert metrics["success_rate"] == 0.0
    assert metrics["avg_response_length"] == 0


def test_evaluate_assistant_empty_response(mock_assistant_config):
    """Test handling of empty responses."""
    # Create a graph that returns empty responses
    empty_graph = MagicMock(spec=StateGraph)
    empty_graph.invoke.return_value = {"answer": ""}
    
    metrics = evaluate_assistant(
        config=mock_assistant_config,
        graph=empty_graph,
        test_questions=["Test question"],
    )
    
    # Should count empty responses as failures
    assert metrics["response_count"] == 0
    assert metrics["success_rate"] == 0.0
    assert metrics["avg_response_length"] == 0 
