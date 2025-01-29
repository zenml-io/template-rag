"""Step to evaluate the RAG assistant's performance."""

import logging
from typing import Annotated

from langgraph.graph.state import CompiledStateGraph
from zenml import step

logger = logging.getLogger(__name__)


@step
def evaluate_assistant(assistant: CompiledStateGraph) -> Annotated[bool, "is_safe"]:
    """Evaluate the assistant's performance with a test question.

    Args:
        assistant: The compiled state graph to evaluate

    Returns:
        bool: True if the assistant generates a non-empty response
    """
    response = assistant.invoke({"question": "What are service connectors?"})["answer"]
    logger.info("Test response: %s", response)
    return bool(response)
