from typing import Annotated

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from materializers.vector_store_materializer import InMemoryVectorStoreMaterializer
from zenml import step

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


@step(output_materializers=InMemoryVectorStoreMaterializer)
def ingest_and_embed(data_path: str) -> Annotated[InMemoryVectorStore, "vector_store"]:
    """Ingest documents and create vector store embeddings.

    Args:
        data_path: Path to the directory containing markdown files.

    Returns:
        InMemoryVectorStore: Vector store containing document embeddings.
    """
    loader = DirectoryLoader(
        data_path,
        glob="**/*.md",
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)
    return vector_store
