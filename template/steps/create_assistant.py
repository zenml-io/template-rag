"""Step to create a RAG assistant."""

import logging
from typing import Annotated

from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, StateGraph
from zenml import step, log_metadata

from config.constants import DEFAULT_LLM_MODEL, DEFAULT_QA_TEMPLATE
from config.models import EMBEDDINGS
from config.store import AssistantConfig, LLMConfig, VectorStoreConfig
from custom_types.state import State
from utils.graph import generate
from utils.vector_store import create_retrieve
from utils.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


@step
def create_assistant(
    vector_store_config: VectorStoreConfig,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.7,
    qa_template: str = DEFAULT_QA_TEMPLATE,
) -> Annotated[AssistantConfig, "assistant_config"]:
    """Create a RAG assistant.

    Args:
        vector_store_config: Configuration for the vector store
        model: Name of the LLM model to use
        temperature: Temperature for LLM generation
        qa_template: Template for QA prompts

    Returns:
        AssistantConfig: Configuration for the RAG assistant
    """
    try:
        # Load vector store
        logger.info(
            "Loading %s vector store from %s",
            vector_store_config.store_type.value,
            vector_store_config.store_path,
        )
        vector_store = VectorStoreFactory.load_store(
            vector_store_config,
            EMBEDDINGS,
        )
        if not vector_store:
            raise ValueError(
                f"Failed to load vector store from {vector_store_config.store_path}"
            )

        # Create LLM config
        llm_config = LLMConfig(
            model=model,
            temperature=temperature,
        )

        # Create graph
        retrieve = create_retrieve(vector_store)
        graph_builder = StateGraph(State)

        # Add nodes and edges in the correct order
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("generate", generate)

        # Add edges to create the flow: retrieve -> generate -> END
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)

        # Set the entry point
        graph_builder.set_entry_point("retrieve")

        # Get graph structure for metadata before compilation
        nodes = ["retrieve", "generate", END]
        edges = [
            "retrieve -> generate",
            "generate -> END",
        ]

        # Compile graph
        graph = graph_builder.compile()

        # Log comprehensive metadata
        log_metadata(
            metadata={
                "assistant": {
                    "vector_store": {
                        "type": vector_store_config.store_type.value,
                        "path": vector_store_config.store_path,
                        "embedding_model": vector_store_config.embedding_model,
                    },
                    "llm": {
                        "model": model,
                        "temperature": temperature,
                        "streaming": llm_config.streaming,
                        "max_tokens": llm_config.max_tokens,
                    },
                    "graph": {
                        "nodes": nodes,
                        "edges": edges,
                    },
                    "prompt": {
                        "template_length": len(qa_template),
                    },
                }
            }
        )

        # Create assistant config
        config = AssistantConfig(
            vector_store=vector_store_config,
            llm=llm_config,
            prompt_templates={"qa": qa_template},
        )

        return config

    except Exception as e:
        logger.error("Failed to create assistant: %s", e)
        raise
