"""Step to create a RAG assistant using a vector store."""

from typing import Annotated, Dict, Optional

from langchain_community.vectorstores import FAISS
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from zenml import step

from config.constants import DEFAULT_LLM_MODEL, DEFAULT_QA_TEMPLATE, DEFAULT_TEMPERATURE
from config.models import EMBEDDINGS
from config.store import AssistantConfig, LLMConfig, VectorStoreConfig
from custom_types.state import State
from utils.graph import generate
from utils.vector_store import create_retrieve


@step
def create_assistant(
    vector_store_config: VectorStoreConfig,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    qa_template: str = DEFAULT_QA_TEMPLATE,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
) -> Annotated[AssistantConfig, "assistant_config"]:
    """Create a RAG assistant using a vector store.

    Args:
        vector_store_config: Configuration for accessing the vector store
        model: Name of the LLM model to use
        temperature: Temperature for LLM generation
        qa_template: Template for QA prompts
        max_tokens: Maximum tokens to generate
        streaming: Whether to stream the response

    Returns:
        Configuration for the RAG assistant
    """
    # Create LLM config and validate
    llm_config = LLMConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )

    # Create assistant config (this will validate prompt template)
    config = AssistantConfig(
        vector_store=vector_store_config,
        llm=llm_config,
        prompt_templates={"qa": qa_template},
    )

    # Load vector store and create graph to verify everything works
    vector_store = FAISS.load_local(
        vector_store_config.store_path,
        EMBEDDINGS,
        allow_dangerous_deserialization=True,
    )
    retrieve = create_retrieve(vector_store)
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    # Test the graph to ensure it works
    response = graph.invoke({"question": "test"})
    if not response.get("answer"):
        raise ValueError("Failed to generate response from graph")

    return config
