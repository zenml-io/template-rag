from zenml import step
from llama_index.core import VectorStoreIndex


@step
def create_assistant(index: str) -> str:
    return "assistant"
