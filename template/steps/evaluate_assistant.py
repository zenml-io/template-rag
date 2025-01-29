import logging
from typing import Annotated

from langgraph.graph.state import CompiledStateGraph
from zenml import step

logger = logging.getLogger(__name__)


@step
def evaluate_assistant(assistant: CompiledStateGraph) -> Annotated[bool, "is_safe"]:
    response = assistant.invoke({"question": "What are service connectors?"})["answer"]
    logger.info(response)
    return bool(response)
