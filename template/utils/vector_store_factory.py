"""Factory module for creating and managing vector stores."""

import logging
from typing import List, Optional, Type

from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from config.constants import VectorStoreType
from config.store import VectorStoreConfig

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Base exception for vector store operations."""

    pass


class UnsupportedStoreError(VectorStoreError):
    """Raised when an unsupported vector store type is requested."""

    pass


class StoreCreationError(VectorStoreError):
    """Raised when vector store creation fails."""

    pass


class StoreLoadError(VectorStoreError):
    """Raised when vector store loading fails."""

    pass


class VectorStoreFactory:
    """Factory class for creating and managing vector stores."""

    _store_classes = {
        VectorStoreType.FAISS: FAISS,
        VectorStoreType.CHROMA: Chroma,
        # Add Pinecone when needed
    }

    @classmethod
    def get_store_class(cls, store_type: VectorStoreType) -> Type[VectorStore]:
        """Get the vector store class for a given type.

        Args:
            store_type: Type of vector store to get

        Returns:
            The vector store class

        Raises:
            UnsupportedStoreError: If store type is not supported
        """
        try:
            if store_type not in cls._store_classes:
                raise UnsupportedStoreError(
                    f"Unsupported vector store type: {store_type}"
                )
            return cls._store_classes[store_type]
        except Exception as e:
            logger.error("Failed to get vector store class: %s", e)
            raise

    @classmethod
    def create_store(
        cls,
        documents: List[Document],
        embeddings: Embeddings,
        config: VectorStoreConfig,
    ) -> VectorStore:
        """Create a new vector store from documents.

        Args:
            documents: List of documents to store
            embeddings: Embeddings model to use
            config: Vector store configuration

        Returns:
            The created vector store

        Raises:
            StoreCreationError: If store creation fails
            UnsupportedStoreError: If store type is not supported
        """
        try:
            store_class = cls.get_store_class(config.store_type)
            logger.info(
                "Creating %s vector store with %d documents",
                config.store_type.value,
                len(documents),
            )

            # Handle store-specific parameters
            kwargs = config.metadata or {}
            if config.store_type == VectorStoreType.FAISS:
                # FAISS doesn't support metric parameter directly
                metric = kwargs.pop("metric", None)
                if metric == "cosine":
                    kwargs["normalize_L2"] = True

            return store_class.from_documents(documents, embeddings, **kwargs)
        except UnsupportedStoreError:
            raise
        except Exception as e:
            logger.error(
                "Failed to create vector store of type %s: %s",
                config.store_type.value,
                e,
            )
            raise StoreCreationError(f"Failed to create vector store: {e}") from e

    @classmethod
    def load_store(
        cls,
        config: VectorStoreConfig,
        embeddings: Embeddings,
    ) -> Optional[VectorStore]:
        """Load an existing vector store.

        Args:
            config: Vector store configuration
            embeddings: Embeddings model to use

        Returns:
            The loaded vector store, or None if it doesn't exist

        Raises:
            StoreLoadError: If store loading fails (except for FileNotFoundError)
            UnsupportedStoreError: If store type is not supported
        """
        try:
            store_class = cls.get_store_class(config.store_type)
            logger.info(
                "Loading %s vector store from %s",
                config.store_type.value,
                config.store_path,
            )
            return store_class.load_local(
                config.store_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except (FileNotFoundError, RuntimeError) as e:
            # FAISS raises RuntimeError for missing files
            if isinstance(e, RuntimeError) and "could not open" in str(e):
                logger.warning("Vector store not found at %s", config.store_path)
                return None
            raise
        except UnsupportedStoreError:
            raise
        except Exception as e:
            logger.error(
                "Failed to load vector store of type %s from %s: %s",
                config.store_type.value,
                config.store_path,
                e,
            )
            raise StoreLoadError(f"Failed to load vector store: {e}") from e
