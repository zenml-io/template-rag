from zenml import step
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine


@step
def create_assistant(index: VectorStoreIndex) -> BaseQueryEngine:
    assistant = index.as_query_engine()
    return assistant
