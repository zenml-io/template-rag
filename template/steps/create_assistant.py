"""Step to create a RAG assistant using a vector store."""

from typing import Annotated

from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from materializers.state_graph_materializer import StateGraphMaterializer
from zenml import step

from custom_types.state import State
from utils.graph import generate
from utils.vector_store import create_retrieve


@step(output_materializers=StateGraphMaterializer)
def create_assistant(
    vector_store: InMemoryVectorStore,
) -> Annotated[CompiledStateGraph, "assistant"]:
    """Create a RAG assistant using a vector store.

    Args:
        vector_store: Vector store containing document embeddings

    Returns:
        A compiled state graph that can be used as a RAG assistant
    """
    retrieve = create_retrieve(vector_store)
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph
