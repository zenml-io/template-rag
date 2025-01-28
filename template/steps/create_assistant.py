from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from zenml import step
from typing import Annotated


@step
def create_assistant(
    index: VectorStoreIndex,
) -> Annotated[BaseQueryEngine, "assistant"]:
    assistant = index.as_query_engine()
    return assistant
