"""Materializer to handle CompiledStateGraph objects."""

import os
from typing import Any, ClassVar, Optional, Tuple, Type

import dill
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from zenml.artifact_stores.base_artifact_store import BaseArtifactStore
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from config.constants import MAX_SEARCH_RESULTS
from config.models import EMBEDDINGS
from custom_types.state import State
from utils.graph import generate
from utils.vector_store import create_retrieve, find_vector_store


class StateGraphMaterializer(BaseMaterializer):
    """Materializer to handle CompiledStateGraph objects."""

    ASSOCIATED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (CompiledStateGraph,)
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.DATA

    def __init__(self, uri: str, artifact_store: Optional[BaseArtifactStore] = None):
        """Initialize the materializer.

        Args:
            uri: The URI where the artifact data is stored.
            artifact_store: The artifact store where the artifact data is stored.
        """
        super().__init__(uri, artifact_store)
        self.store_path = os.path.join(self.uri, "state_graph.dill")

    def load(self, data_type: Type[Any]) -> CompiledStateGraph:
        """Read from artifact store and reconstruct the graph.

        Args:
            data_type: The type of the data to read.

        Returns:
            CompiledStateGraph: The reconstructed state graph.
        """
        with self.artifact_store.open(self.store_path, "rb") as f:
            data = dill.load(f)

        # Create a FAISS vector store with the documents
        vector_store = FAISS.from_texts(
            texts=data["texts"], embedding=EMBEDDINGS, metadatas=data["metadatas"]
        )

        # Build the graph
        retrieve = create_retrieve(vector_store)
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        return graph

    def save(self, graph: CompiledStateGraph) -> None:
        """Write essential components to artifact store.

        Args:
            graph: The state graph to write.
        """
        # Extract the vector store from the graph's state
        vector_store = find_vector_store(graph)

        if vector_store is None:
            raise ValueError(
                "Could not extract vector store from graph. This is required for serialization."
            )

        # Extract just the documents and their metadata
        documents = []
        for doc in vector_store.similarity_search("", k=MAX_SEARCH_RESULTS):
            documents.append(doc)

        # Store only the essential data
        data = {
            "texts": [doc.page_content for doc in documents],
            "metadatas": [doc.metadata for doc in documents],
        }

        with self.artifact_store.open(self.store_path, "wb") as f:
            dill.dump(data, f)
