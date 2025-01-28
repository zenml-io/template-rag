from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from zenml import step
from typing import Annotated


@step
def ingest_and_embed(data_path: str) -> Annotated[VectorStoreIndex, "vector_index"]:
    documents = SimpleDirectoryReader(data_path).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index
