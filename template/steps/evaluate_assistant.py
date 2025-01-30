"""Step to evaluate the RAG assistant's performance."""

import logging
from typing import Annotated, Dict, List, Union, Any

from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from zenml import step, log_metadata

from config.models import EMBEDDINGS
from config.store import AssistantConfig
from custom_types.state import State
from utils.graph import generate
from utils.vector_store import create_retrieve
from utils.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_TEST_QUESTIONS = [
    "What are service connectors?",
    "How do I deploy ZenML?",
]


@step
def evaluate_assistant(
    graph: Any,
    test_questions: List[str] = [
        "What are service connectors?",
        "How do I deploy ZenML?"
    ],
    llm_config: Dict[str, Any] = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7
    }
) -> Dict[str, Any]:
    """Evaluate the assistant by asking test questions and measuring response quality."""
    try:
        logger.info("Starting evaluation with graph type: %s", type(graph))
        logger.debug("Graph attributes: %s", dir(graph))
        logger.debug("Graph nodes: %s", graph.nodes)
        logger.debug("Graph edges: %s", graph.edges)
        logger.debug("Graph state schema: %s", getattr(graph, 'state_schema', 'No schema found'))
        logger.debug("Graph input type: %s", getattr(graph, 'input', None))
        logger.debug("Graph output type: %s", getattr(graph, 'output', None))
        logger.debug("Graph dict: %s", vars(graph))

        # Log details about each node
        for node_name, node in graph.nodes.items():
            logger.debug(f"Node {node_name} details:")
            logger.debug("  Type: %s", type(node))
            logger.debug("  Attributes: %s", dir(node))
            logger.debug("  Dict: %s", getattr(node, '__dict__', 'No dict'))
            if hasattr(node, 'runnable'):
                logger.debug("  Runnable type: %s", type(node.runnable))
                logger.debug("  Runnable attributes: %s", dir(node.runnable))

        # Configure evaluation
        logger.info("Configuring evaluation with %d test questions", len(test_questions))
        logger.debug("Test questions: %s", test_questions)
        logger.debug("LLM configuration: model=%s, temperature=%f", 
                    llm_config.get('model'), llm_config.get('temperature'))
        logger.debug("Full LLM config: %s", llm_config)

        # Process each test question
        responses = []
        for i, question in enumerate(test_questions, 1):
            logger.info("Processing question %d/%d: %s", i, len(test_questions), question)
            try:
                # Prepare input state
                logger.debug("Preparing graph input state for question")
                input_state = {"question": question}
                logger.debug("Input state: %s", input_state)

                # Invoke graph
                logger.debug("Invoking graph with input state")
                if hasattr(graph, 'invoke'):
                    result = graph.invoke(input_state)
                    logger.debug("Graph result: %s", result)
                    response = result.get('answer', '')
                    logger.debug("Extracted response: %s", response)
                    responses.append(response)
                else:
                    logger.error("Graph does not have invoke method")
                    logger.debug("Graph state at failure: %s", {
                        'nodes': graph.nodes,
                        'edges': graph.edges,
                        'state_schema': getattr(graph, 'state_schema', 'N/A')
                    })
                    logger.debug("Graph dict at failure: %s", vars(graph))
                    raise ValueError("Graph does not have invoke method")

            except Exception as e:
                logger.error("Failed to get response for question '%s': %s", question, str(e))
                logger.debug("Graph state at failure: %s", {
                    'nodes': graph.nodes,
                    'edges': graph.edges,
                    'state_schema': getattr(graph, 'state_schema', 'N/A')
                })
                logger.debug("Graph dict at failure: %s", vars(graph))
                responses.append('')

        # Calculate evaluation metrics
        logger.info("Calculating evaluation metrics")
        non_empty_responses = [r for r in responses if r.strip()]
        response_lengths = [len(r.split()) for r in responses]
        
        metrics = {
            'response_count': len(non_empty_responses),
            'avg_response_length': sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            'success_rate': len(non_empty_responses) / len(test_questions)
        }
        
        logger.debug("Non-empty responses: %d/%d", len(non_empty_responses), len(test_questions))
        logger.debug("Response lengths: %s", response_lengths)
        logger.debug("Evaluation metrics: %s", metrics)

        # Log final summary
        logger.info("Logging final evaluation summary")
        return metrics

    except Exception as e:
        logger.error("Failed to evaluate assistant: %s", str(e))
        raise
