import logging
import os
import asyncio
import nest_asyncio
from contextlib import contextmanager

from llama_index.core.base.base_query_engine import BaseQueryEngine
from traceloop.sdk import Traceloop
from zenml import step
from utils.workflows import RAGWorkflow

logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

TRACELOOP_API_KEY = os.getenv("TRACELOOP_API_KEY")


@contextmanager
def initialize_traceloop():
    """Initialize Traceloop in a context manager.

    This ensures proper initialization and cleanup of Traceloop context.
    """
    try:
        Traceloop.init(
            api_key=TRACELOOP_API_KEY,
            disable_batch=True,  # Disable batching for immediate trace visibility
        )
        yield
    finally:
        pass


async def _evaluate_assistant_async(assistant: RAGWorkflow) -> bool:
    """Evaluate the assistant with Traceloop instrumentation.

    Args:
        assistant: The RAG workflow to evaluate

    Returns:
        bool: Success status of the evaluation
    """
    with initialize_traceloop():
        response = await assistant.run(query="What are ZenML service connectors?")
        # Get the response text directly from the response object
        if hasattr(response, "source_nodes"):
            # Extract text from source nodes if available
            texts = [node.node.text for node in response.source_nodes]
            logger.info("\n".join(texts))
        else:
            logger.info("Response received but no source nodes available")
        return bool(response)


@step(enable_cache=False)
def evaluate_assistant(assistant: RAGWorkflow) -> bool:
    """ZenML step to evaluate the assistant.

    Args:
        assistant: The RAG workflow to evaluate

    Returns:
        bool: Success status of the evaluation
    """
    return asyncio.run(_evaluate_assistant_async(assistant))
