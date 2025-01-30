"""Utility functions for serializing and deserializing vector stores and graphs."""

from typing import Any, Dict

import dill
from config.constants import MAX_SEARCH_RESULTS
from langchain_core.vectorstores import VectorStore
from zenml.artifact_stores.base_artifact_store import BaseArtifactStore

from utils.vector_store import extract_store_data


def save_vector_store(
    vector_store: VectorStore,
    store_path: str,
    artifact_store: BaseArtifactStore,
) -> None:
    """Save a vector store to the artifact store.

    Args:
        vector_store: The vector store to save
        store_path: Path where to save the vector store
        artifact_store: The artifact store to save to
    """
    texts, metadatas, embeddings = extract_store_data(
        vector_store, k=MAX_SEARCH_RESULTS
    )
    data = {
        "texts": texts,
        "metadatas": metadatas,
        "embeddings": embeddings,
    }

    with artifact_store.open(store_path, "wb") as f:
        dill.dump(data, f)


def load_vector_store_data(
    store_path: str,
    artifact_store: BaseArtifactStore,
) -> Dict[str, Any]:
    """Load vector store data from the artifact store.

    Args:
        store_path: Path where the vector store is saved
        artifact_store: The artifact store to load from

    Returns:
        Dictionary containing the vector store data
    """
    with artifact_store.open(store_path, "rb") as f:
        return dill.load(f)
