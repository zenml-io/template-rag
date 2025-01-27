from zenml import step
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


@step
def ingest_and_embed(data_path: str) -> str:
    documents = SimpleDirectoryReader(data_path).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return "index"
