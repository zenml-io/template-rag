"""Utility functions for working with vector stores."""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from config.constants import MAX_SEARCH_RESULTS


def create_retrieve(vector_store: VectorStore, k: int = MAX_SEARCH_RESULTS):
    """Create a retrieval function for the RAG pipeline.

    Args:
        vector_store: The vector store to retrieve documents from
        k: Number of documents to retrieve (default: 4)

    Returns:
        A function that takes a state dict and returns retrieved documents
    """

    def retrieve(state: Dict[str, Any]) -> Dict[str, List[Document]]:
        retrieved_docs = vector_store.similarity_search(state["question"], k=k)
        return {"context": retrieved_docs}

    return retrieve


def extract_store_data(
    vector_store: VectorStore, k: Optional[int] = None
) -> Tuple[List[str], List[Dict[str, Any]], List[List[float]]]:
    """Extract texts, metadata, and embeddings from a vector store.

    Args:
        vector_store: The vector store to extract data from
        k: Maximum number of documents to extract (default: None = all documents)

    Returns:
        Tuple of (texts, metadata, embeddings)
    """
    results = vector_store.similarity_search_with_score("", k=k or float("inf"))

    texts = []
    metadatas = []
    embeddings = []

    for doc, _ in results:
        texts.append(doc.page_content)
        metadatas.append(doc.metadata)
        embeddings.append(vector_store.embeddings.embed_query(doc.page_content))

    return texts, metadatas, embeddings


def find_vector_store(obj: Any, visited: set = None) -> Any:
    """Recursively search for a vector store in an object's attributes.

    Args:
        obj: The object to search
        visited: Set of visited object IDs to avoid circular references

    Returns:
        The vector store if found, None otherwise
    """
    if visited is None:
        visited = set()

    # Avoid circular references
    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    # Check if this object has similarity_search method
    if hasattr(obj, "similarity_search"):
        return obj

    # If it's a function, check its closure
    if isinstance(obj, type(lambda: None)):
        if hasattr(obj, "__closure__") and obj.__closure__:
            for cell in obj.__closure__:
                result = find_vector_store(cell.cell_contents, visited)
                if result:
                    return result

    # If it's a dictionary, check its values
    elif isinstance(obj, dict):
        for value in obj.values():
            result = find_vector_store(value, visited)
            if result:
                return result

    # If it has __dict__, check all its attributes
    elif hasattr(obj, "__dict__"):
        for value in obj.__dict__.values():
            result = find_vector_store(value, visited)
            if result:
                return result

    # If it's a list or tuple, check its elements
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            result = find_vector_store(item, visited)
            if result:
                return result

    return None
