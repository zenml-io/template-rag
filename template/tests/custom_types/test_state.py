"""Tests for custom state type."""

from typing import List, cast

import pytest
from langchain_core.documents import Document

from custom_types.state import State


@pytest.fixture
def sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    return [
        Document(page_content="test content 1", metadata={"source": "test1"}),
        Document(page_content="test content 2", metadata={"source": "test2"}),
    ]


def test_state_creation():
    """Test State dictionary creation."""
    # Test minimal state
    state: State = {
        "question": "",
        "context": [],
        "answer": "",
    }
    assert isinstance(state, dict)
    assert state["question"] == ""
    assert state["context"] == []
    assert state["answer"] == ""

    # Test state with values
    state: State = {
        "question": "test question",
        "context": [Document(page_content="test", metadata={})],
        "answer": "test answer",
    }
    assert state["question"] == "test question"
    assert len(state["context"]) == 1
    assert state["answer"] == "test answer"


def test_state_runtime_behavior(sample_documents):
    """Test State runtime behavior.

    Note: TypedDict only provides static type checking, not runtime checking.
    These tests verify the expected runtime behavior while maintaining type safety
    in the static type checker.
    """
    # Test valid state
    state: State = {
        "question": "test question",
        "context": sample_documents,
        "answer": "test answer",
    }
    assert isinstance(state["question"], str)
    assert isinstance(state["context"], list)
    assert all(isinstance(doc, Document) for doc in state["context"])
    assert isinstance(state["answer"], str)

    # Runtime behavior: invalid types are allowed (but caught by static type checker)
    runtime_state = cast(
        State,
        {
            "question": 123,  # type: ignore  # Would be caught by static type checker
            "context": sample_documents,
            "answer": "test",
        },
    )
    assert isinstance(runtime_state, dict)

    # Runtime behavior: missing fields are allowed (but caught by static type checker)
    incomplete_state = cast(
        State,
        {  # type: ignore  # Would be caught by static type checker
            "question": "test",
            "answer": "test",
        },
    )
    assert isinstance(incomplete_state, dict)


def test_state_static_typing():
    """Demonstrate static typing expectations.

    Note: These are not actual tests, but examples of what the static
    type checker would catch. The comments show what would fail type checking.
    """
    # Valid state
    state: State = {
        "question": "test",
        "context": [],
        "answer": "test",
    }

    # The following would fail static type checking:
    # state: State = {
    #     "question": 123,  # Type error: expected str
    #     "context": [],
    #     "answer": "test",
    # }

    # state: State = {
    #     "question": "test",
    #     # Missing required field 'context'
    #     "answer": "test",
    # }

    assert isinstance(state, dict)


def test_state_mutability(sample_documents):
    """Test State dictionary mutability."""
    state: State = {
        "question": "test question",
        "context": sample_documents.copy(),
        "answer": "test answer",
    }

    # State is a regular dict, so it should be mutable
    state["question"] = "new question"
    assert state["question"] == "new question"

    state["context"] = []
    assert state["context"] == []

    state["answer"] = "new answer"
    assert state["answer"] == "new answer"


def test_state_as_function_arg(sample_documents):
    """Test using State as a function argument."""

    def process_state(state: State) -> str:
        """Test function that processes a state."""
        return f"Q: {state['question']}, A: {state['answer']}"

    state: State = {
        "question": "test",
        "context": sample_documents,
        "answer": "answer",
    }

    result = process_state(state)
    assert result == "Q: test, A: answer"


def test_state_dict_operations(sample_documents):
    """Test standard dictionary operations on State."""
    state: State = {
        "question": "test",
        "context": sample_documents,
        "answer": "answer",
    }

    # Test keys
    assert set(state.keys()) == {"question", "context", "answer"}

    # Test values access
    assert all(isinstance(v, (str, list)) for v in state.values())

    # Test items iteration
    for key, value in state.items():
        assert key in ["question", "context", "answer"]
        assert isinstance(value, (str, list))
