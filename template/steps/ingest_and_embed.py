"""Step to ingest documents and create embeddings."""

import logging
from typing import Annotated

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from zenml import step

from config.constants import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_STORE_PATH,
    VectorStoreType,
)
from config.models import EMBEDDINGS
from config.store import VectorStoreConfig
from utils.vector_store_factory import (
    VectorStoreFactory,
    VectorStoreError,
    UnsupportedStoreError,
    StoreCreationError,
)

logger = logging.getLogger(__name__)


@step
def ingest_and_embed(
    data_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    store_path: str = DEFAULT_VECTOR_STORE_PATH,
    store_type: VectorStoreType = VectorStoreType.FAISS,
) -> Annotated[VectorStoreConfig, "vector_store_config"]:
    """Ingest documents and create vector store embeddings.

    Args:
        data_path: Path to the directory containing markdown files.
        chunk_size: Size of text chunks for splitting documents.
        chunk_overlap: Overlap between text chunks.
        store_path: Path where to save the vector store.
        store_type: Type of vector store to use.

    Returns:
        VectorStoreConfig: Configuration for accessing the vector store.

    Raises:
        UnsupportedStoreError: If the specified store type is not supported
        StoreCreationError: If vector store creation fails
        FileNotFoundError: If data path doesn't exist or is empty
        ValueError: If configuration validation fails
    """
    try:
        # Create and validate config first to fail fast if settings are invalid
        config = VectorStoreConfig(
            store_type=store_type,
            store_path=store_path,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            documents_path=data_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Load and process documents
        logger.info("Loading documents from %s", data_path)
        loader = DirectoryLoader(
            data_path,
            glob="**/*.md",
        )
        docs = loader.load()
        if not docs:
            raise FileNotFoundError(f"No markdown files found in {data_path}")

        logger.info("Splitting %d documents into chunks", len(docs))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        all_splits = text_splitter.split_documents(docs)
        logger.info("Created %d text chunks", len(all_splits))

        # Create and save vector store using factory
        vector_store = VectorStoreFactory.create_store(
            documents=all_splits,
            embeddings=EMBEDDINGS,
            config=config,
        )
        logger.info("Saving vector store to %s", config.store_path)
        vector_store.save_local(config.store_path)

        return config

    except UnsupportedStoreError:
        logger.error("Unsupported vector store type: %s", store_type)
        raise
    except StoreCreationError:
        logger.error("Failed to create vector store at %s", store_path)
        raise
    except FileNotFoundError as e:
        logger.error("Document loading failed: %s", e)
        raise
    except ValueError as e:
        logger.error("Configuration validation failed: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during ingestion: %s", e)
        raise
