import os
from typing import Any, ClassVar, Optional, Tuple, Type

import dill
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from zenml.artifact_stores.base_artifact_store import BaseArtifactStore
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer


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

        # Recreate vector store from scratch
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = InMemoryVectorStore(embeddings)

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
        # Get all documents through similarity search
        # This will return all documents in the store
        results = vector_store.similarity_search_with_score("", k=10000)

        # Extract texts, embeddings and metadata
        texts = []
        metadatas = []
        embeddings = []

        for doc, _ in results:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
            # Get embedding for this text
            embeddings.append(vector_store.embeddings.embed_query(doc.page_content))

        data = {"texts": texts, "embeddings": embeddings, "metadatas": metadatas}

        with self.artifact_store.open(self.store_path, "wb") as f:
            dill.dump(data, f)
