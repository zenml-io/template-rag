"""Step to create a RAG assistant."""

import logging
from typing import Annotated, Tuple

from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from zenml import step, log_metadata

from config.constants import DEFAULT_LLM_MODEL, DEFAULT_QA_TEMPLATE
from config.models import EMBEDDINGS
from config.store import AssistantConfig, LLMConfig, VectorStoreConfig
from custom_types.state import State
from utils.graph import generate
from utils.vector_store import create_retrieve
from utils.vector_store_factory import VectorStoreFactory
from materializers.langgraph_materializer import LangGraphMaterializer
from materializers.assistant_config_materializer import AssistantConfigMaterializer

logger = logging.getLogger(__name__)


@step(
    output_materializers={
        "assistant_config": AssistantConfigMaterializer,
        "graph": LangGraphMaterializer
    }
)
def create_assistant(
    vector_store_config: VectorStoreConfig,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.7,
    qa_template: str = DEFAULT_QA_TEMPLATE,
) -> Tuple[Annotated[AssistantConfig, "assistant_config"], Annotated[CompiledStateGraph, "graph"]]:
    """Create a RAG assistant.

    Args:
        vector_store_config: Configuration for the vector store
        model: Name of the LLM model to use
        temperature: Temperature for LLM generation
        qa_template: Template for QA prompts

    Returns:
        Tuple[AssistantConfig, CompiledStateGraph]: Configuration and graph for the RAG assistant
    """
    try:
        # Load vector store
        logger.info(
            "Loading %s vector store from %s",
            vector_store_config.store_type.value,
            vector_store_config.store_path,
        )
        logger.debug("Vector store config: %s", vars(vector_store_config))
        
        vector_store = VectorStoreFactory.load_store(
            vector_store_config,
            EMBEDDINGS,
        )
        if not vector_store:
            raise ValueError(
                f"Failed to load vector store from {vector_store_config.store_path}"
            )
        
        logger.debug("Vector store details: %s", {
            'type': type(vector_store),
            'attributes': dir(vector_store),
            'dict': vars(vector_store) if hasattr(vector_store, '__dict__') else 'No dict'
        })

        # Create LLM config
        llm_config = LLMConfig(
            model=model,
            temperature=temperature,
        )
        logger.debug("Created LLM config: %s", vars(llm_config))

        # Create graph
        logger.debug("Creating retrieve function")
        retrieve = create_retrieve(vector_store)
        logger.debug("Retrieve function details: %s", {
            'type': type(retrieve),
            'attributes': dir(retrieve),
            'callable': callable(retrieve)
        })
        
        logger.debug("Creating StateGraph with State schema")
        graph_builder = StateGraph(State)
        logger.debug("Graph builder details: %s", {
            'type': type(graph_builder),
            'attributes': dir(graph_builder),
            'dict': vars(graph_builder)
        })

        # Add nodes and edges in the correct order
        logger.debug("Adding retrieve node")
        graph_builder.add_node("retrieve", retrieve)
        logger.debug("Adding generate node")
        graph_builder.add_node("generate", generate)
        logger.debug("Current nodes: %s", list(graph_builder.nodes.keys()))
        
        # Detailed inspection of nodes
        for node_name, node in graph_builder.nodes.items():
            logger.debug("Node %s details:", node_name)
            logger.debug("  Type: %s", type(node))
            logger.debug("  Attributes: %s", dir(node))
            logger.debug("  Dict: %s", vars(node) if hasattr(node, '__dict__') else 'No dict')

        # Add edges to create the flow: retrieve -> generate -> END
        logger.debug("Adding edge: retrieve -> generate")
        graph_builder.add_edge("retrieve", "generate")
        logger.debug("Adding edge: generate -> END")
        graph_builder.add_edge("generate", END)
        logger.debug("Current edges: %s", getattr(graph_builder, 'edges', 'No edges found'))

        # Set the entry point
        logger.debug("Setting entry point to 'retrieve'")
        graph_builder.set_entry_point("retrieve")
        logger.debug("Entry point set: %s", getattr(graph_builder, 'entry_point', 'No entry point found'))

        # Get graph structure for metadata before compilation
        nodes = ["retrieve", "generate", "__end__"]
        edges = [
            "retrieve -> generate",
            "generate -> __end__",
        ]

        # Compile graph
        logger.info("Compiling graph with nodes: %s", nodes)
        logger.info("Graph edges: %s", edges)
        logger.info("Graph entry point: retrieve")
        
        logger.debug("Graph builder state before compilation: %s", {
            'nodes': list(graph_builder.nodes.keys()),
            'edges': getattr(graph_builder, 'edges', 'No edges found'),
            'entry_point': getattr(graph_builder, 'entry_point', 'No entry point found'),
            'state_schema': getattr(graph_builder, 'state_schema', None),
            'input_type': getattr(graph_builder, 'input', None),
            'output_type': getattr(graph_builder, 'output', None),
            'dict': vars(graph_builder)
        })
        
        graph = graph_builder.compile()
        
        # Verify graph structure after compilation
        logger.info("Compiled graph nodes: %s", list(graph.nodes.keys()))
        logger.info("Compiled graph edges: %s", getattr(graph, 'edges', 'No edges found'))
        logger.info("Compiled graph state schema: %s", getattr(graph, 'state_schema', None))
        
        logger.debug("Compiled graph details: %s", {
            'type': type(graph),
            'attributes': dir(graph),
            'nodes': list(graph.nodes.keys()),
            'edges': getattr(graph, 'edges', 'No edges found'),
            'state_schema': getattr(graph, 'state_schema', None),
            'input_type': getattr(graph, 'input', None),
            'output_type': getattr(graph, 'output', None),
            'dict': vars(graph)
        })
        
        # Detailed inspection of compiled nodes
        for node_name, node in graph.nodes.items():
            logger.debug("Compiled node %s details:", node_name)
            logger.debug("  Type: %s", type(node))
            logger.debug("  Attributes: %s", dir(node))
            logger.debug("  Dict: %s", vars(node) if hasattr(node, '__dict__') else 'No dict')
            if hasattr(node, 'runnable'):
                logger.debug("  Runnable type: %s", type(node.runnable))
                logger.debug("  Runnable attributes: %s", dir(node.runnable))
            if hasattr(node, 'writers'):
                logger.debug("  Writers: %s", [type(w) for w in node.writers])
                for i, writer in enumerate(node.writers):
                    logger.debug("  Writer %d details:", i)
                    logger.debug("    Type: %s", type(writer))
                    logger.debug("    Attributes: %s", dir(writer))
                    logger.debug("    Dict: %s", vars(writer) if hasattr(writer, '__dict__') else 'No dict')
        
        # Restore edges after compilation
        graph.edges = graph_builder.edges
        logger.info("Restored edges after compilation: %s", graph.edges)

        # Log comprehensive metadata
        metadata = {
            "assistant": {
                "vector_store": {
                    "type": vector_store_config.store_type.value,
                    "path": vector_store_config.store_path,
                    "embedding_model": vector_store_config.embedding_model,
                    "config": vars(vector_store_config),
                },
                "llm": {
                    "model": model,
                    "temperature": temperature,
                    "streaming": llm_config.streaming,
                    "max_tokens": llm_config.max_tokens,
                    "config": vars(llm_config),
                },
                "graph": {
                    "nodes": nodes,
                    "edges": edges,
                    "compiled_type": str(type(graph)),
                    "node_types": {name: str(type(node)) for name, node in graph.nodes.items()},
                },
                "prompt": {
                    "template_length": len(qa_template),
                    "template": qa_template,
                },
            }
        }
        logger.debug("Logging metadata: %s", metadata)
        log_metadata(metadata=metadata)

        # Create assistant config
        config = AssistantConfig(
            vector_store=vector_store_config,
            llm=llm_config,
            prompt_templates={"qa": qa_template},
        )
        logger.debug("Created assistant config: %s", vars(config))

        return config, graph

    except Exception as e:
        logger.error("Failed to create assistant: %s", e, exc_info=True)
        raise
