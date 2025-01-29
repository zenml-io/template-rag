"""Step to ingest documents and create embeddings."""

from typing import Annotated

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from zenml import step

from config.constants import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_STORE_PATH,
)
from config.models import EMBEDDINGS
from config.store import VectorStoreConfig


@step
def ingest_and_embed(
    data_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    store_path: str = DEFAULT_VECTOR_STORE_PATH,
) -> Annotated[VectorStoreConfig, "vector_store_config"]:
    """Ingest documents and create vector store embeddings.

    Args:
        data_path: Path to the directory containing markdown files.
        chunk_size: Size of text chunks for splitting documents.
        chunk_overlap: Overlap between text chunks.
        store_path: Path where to save the vector store.

    Returns:
        VectorStoreConfig: Configuration for accessing the vector store.
    """
    # Create and validate config first to fail fast if settings are invalid
    config = VectorStoreConfig(
        store_type="faiss",  # Currently only supporting FAISS
        store_path=store_path,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        documents_path=data_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Load and process documents
    loader = DirectoryLoader(
        data_path,
        glob="**/*.md",
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    all_splits = text_splitter.split_documents(docs)

    # Create and populate vector store
    vector_store = FAISS.from_documents(all_splits, EMBEDDINGS)

    # Save vector store to disk/storage
    vector_store.save_local(config.store_path)

    return config
