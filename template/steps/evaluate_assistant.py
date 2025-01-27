import logging
import os

from llama_index.core.base.base_query_engine import BaseQueryEngine
from traceloop.sdk import Traceloop
from zenml import step

logger = logging.getLogger(__name__)

TRACELOOP_API_KEY = os.getenv("TRACELOOP_API_KEY")


@step
def evaluate_assistant(assistant: BaseQueryEngine) -> bool:
    Traceloop.init(api_key=TRACELOOP_API_KEY)
    response = assistant.query("What are ZenML service connectors?")
    logger.info(response)
    return bool(response)
