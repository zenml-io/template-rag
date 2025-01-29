from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from zenml import step
from utils.workflows import RAGWorkflow
from llama_index.core.workflow import Workflow


@step
def create_assistant(index: VectorStoreIndex) -> RAGWorkflow:
    w = RAGWorkflow(index=index)
    return w
