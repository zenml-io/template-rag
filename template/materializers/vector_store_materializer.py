"""Materializer to handle InMemoryVectorStore objects."""

import os
from typing import Any, ClassVar, Optional, Tuple, Type

import dill
from langchain_core.vectorstores import InMemoryVectorStore
from zenml.artifact_stores.base_artifact_store import BaseArtifactStore
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from config.constants import MAX_SEARCH_RESULTS
from config.models import EMBEDDINGS
from utils.vector_store import extract_store_data


class InMemoryVectorStoreMaterializer(BaseMaterializer):
    """Materializer to handle InMemoryVectorStore objects."""

    ASSOCIATED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (InMemoryVectorStore,)
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.DATA

    def __init__(self, uri: str, artifact_store: Optional[BaseArtifactStore] = None):
        """Initialize the materializer.

        Args:
            uri: The URI where the artifact data is stored.
            artifact_store: The artifact store where the artifact data is stored.
        """
        super().__init__(uri, artifact_store)
        self.store_path = os.path.join(self.uri, "documents.dill")

    def load(self, data_type: Type[Any]) -> InMemoryVectorStore:
        """Read from artifact store.

        Args:
            data_type: The type of the data to read.

        Returns:
            InMemoryVectorStore: The deserialized vector store.
        """
        with self.artifact_store.open(self.store_path, "rb") as f:
            data = dill.load(f)

        # Create vector store from scratch
        vector_store = InMemoryVectorStore(EMBEDDINGS)

        # Add the documents back with their embeddings
        vector_store.add_texts(
            texts=data["texts"],
            metadatas=data["metadatas"],
            embeddings=data["embeddings"],
        )
        return vector_store

    def save(self, vector_store: InMemoryVectorStore) -> None:
        """Write to artifact store.

        Args:
            vector_store: The vector store to write.
        """
        # Extract texts, embeddings and metadata
        texts, metadatas, embeddings = extract_store_data(
            vector_store, k=MAX_SEARCH_RESULTS
        )
        data = {"texts": texts, "embeddings": embeddings, "metadatas": metadatas}

        with self.artifact_store.open(self.store_path, "wb") as f:
            dill.dump(data, f)
