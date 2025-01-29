"""Utility functions for working with LangGraph state graphs."""

from typing import Dict

from langchain import hub

from config.models import LLM
from custom_types.state import State


def generate(state: State) -> Dict[str, str]:
    """Generate an answer based on the question and context.

    Args:
        state: The current state containing question and context

    Returns:
        Dict containing the generated answer
    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = hub.pull("rlm/rag-prompt")
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = LLM.invoke(messages)
    return {"answer": response.content}
