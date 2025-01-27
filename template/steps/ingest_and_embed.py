from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from zenml import step


@step
def ingest_and_embed(data_path: str) -> VectorStoreIndex:
    documents = SimpleDirectoryReader(data_path).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index
