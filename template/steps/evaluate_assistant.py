from zenml import step
from traceloop.sdk import Traceloop
from llama_index.core.base.base_query_engine import BaseQueryEngine
import logging
import os

logger = logging.getLogger(__name__)

TRACELOOP_API_KEY = os.getenv("TRACELOOP_API_KEY")


@step
def evaluate_assistant(assistant: BaseQueryEngine) -> bool:
    Traceloop.init(api_key=TRACELOOP_API_KEY)
    response = assistant.query("What are ZenML service connectors?")
    logger.info(response)
    return bool(response)
