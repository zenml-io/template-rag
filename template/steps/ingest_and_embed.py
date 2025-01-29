"""Step to ingest and embed documents."""

import logging
from pathlib import Path
from typing import Annotated

from zenml import step, log_metadata
from zenml.metadata.metadata_types import StorageSize

from config.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_VECTOR_STORE_PATH,
    DEFAULT_EMBEDDING_MODEL,
    VectorStoreType,
)
from config.models import EMBEDDINGS
from config.store import VectorStoreConfig
from utils.document_loader import load_documents
from utils.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


@step
def ingest_and_embed(
    docs_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    store_path: str = DEFAULT_VECTOR_STORE_PATH,
    store_type: VectorStoreType = VectorStoreType.FAISS,
) -> Annotated[VectorStoreConfig, "vector_store_config"]:
    """Ingest and embed documents.

    Args:
        docs_path: Path to the documents to ingest
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        store_path: Path where to save the vector store
        store_type: Type of vector store to use

    Returns:
        VectorStoreConfig: Configuration for the vector store
    """
    try:
        # Load and process documents
        logger.info("Loading documents from %s", docs_path)
        chunks = load_documents(docs_path, chunk_size, chunk_overlap)
        if not chunks:
            raise ValueError(f"No documents found in {docs_path}")

        # Calculate document statistics
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        source_files = set(Path(chunk.metadata["source"]) for chunk in chunks)
        file_types = list(set(p.suffix for p in source_files))

        # Log document processing metadata
        log_metadata(
            metadata={
                "document_stats": {
                    "num_source_files": len(source_files),
                    "total_content_size": StorageSize(total_chars),
                    "avg_chunk_size": StorageSize(total_chars // len(chunks)),
                    "file_types": file_types,
                }
            }
        )

        # Log chunking metadata
        log_metadata(
            metadata={
                "chunking_stats": {
                    "num_chunks": len(chunks),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
            }
        )

        # Create vector store config
        config = VectorStoreConfig(
            store_type=store_type,
            store_path=store_path,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            documents_path=docs_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Create vector store
        vector_store = VectorStoreFactory.create_store(
            documents=chunks,
            embeddings=EMBEDDINGS,
            config=config,
        )
        vector_store.save_local(store_path)

        # Log embedding metadata
        log_metadata(
            metadata={
                "embedding_stats": {
                    "embedding_model": config.embedding_model,
                    "vector_store_type": config.store_type.value,
                    "num_vectors": len(chunks),
                    "store_path": store_path,
                }
            }
        )

        return config

    except Exception as e:
        logger.error("Failed to ingest and embed documents: %s", e)
        raise
