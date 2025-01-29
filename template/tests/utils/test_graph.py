"""Tests for graph utility functions."""

import pytest
from langchain_core.documents import Document
from unittest.mock import patch, MagicMock

from utils.graph import generate
from custom_types.state import State


def test_generate_answer(mock_llm):
    """Test generating an answer from question and context."""
    # Create test state
    state: State = {
        "question": "What is the meaning of life?",
        "context": [
            Document(
                page_content="The meaning of life is 42.",
                metadata={"source": "test.txt"},
            ),
            Document(
                page_content="Life's meaning is subjective.",
                metadata={"source": "test2.txt"},
            ),
        ],
    }

    # Mock the hub.pull to return a prompt template
    mock_prompt = MagicMock()
    mock_prompt.invoke.return_value = "test messages"

    with patch("utils.graph.hub.pull", return_value=mock_prompt):
        with patch("utils.graph.LLM", mock_llm):
            result = generate(state)

    # Verify the result structure
    assert isinstance(result, dict)
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert result["answer"] == "This is a test response"  # From mock_llm fixture


def test_generate_empty_context():
    """Test generating an answer with empty context."""
    state: State = {"question": "What is the meaning of life?", "context": []}

    # Mock the hub.pull to return a prompt template
    mock_prompt = MagicMock()
    mock_prompt.invoke.return_value = "test messages"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="No context available")

    with patch("utils.graph.hub.pull", return_value=mock_prompt):
        with patch("utils.graph.LLM", mock_llm):
            result = generate(state)

    assert result["answer"] == "No context available"


def test_generate_long_context():
    """Test generating an answer with a long context."""
    # Create a long context
    long_docs = [
        Document(
            page_content=f"This is test document {i} with some content.",
            metadata={"source": f"test{i}.txt"},
        )
        for i in range(10)
    ]

    state: State = {"question": "What is in the documents?", "context": long_docs}

    # Mock the hub.pull to return a prompt template
    mock_prompt = MagicMock()
    mock_prompt.invoke.return_value = "test messages"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Summary of long context")

    with patch("utils.graph.hub.pull", return_value=mock_prompt):
        with patch("utils.graph.LLM", mock_llm):
            result = generate(state)

    assert result["answer"] == "Summary of long context"
    # Verify that all documents were included in the context
    mock_prompt.invoke.assert_called_once()
    context_arg = mock_prompt.invoke.call_args[0][0]["context"]
    assert all(f"test document {i}" in context_arg for i in range(10))
