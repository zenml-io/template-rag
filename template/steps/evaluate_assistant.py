"""Step to evaluate the RAG assistant's performance."""

import logging
from typing import Annotated, Dict, List

from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, StateGraph
from zenml import step, log_metadata

from config.models import EMBEDDINGS
from config.store import AssistantConfig
from custom_types.state import State
from utils.graph import generate
from utils.vector_store import create_retrieve
from utils.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)

DEFAULT_TEST_QUESTIONS = [
    "What are service connectors?",
    "How do I deploy ZenML?",
]


@step
def evaluate_assistant(
    config: AssistantConfig,
    test_questions: List[str] = DEFAULT_TEST_QUESTIONS,
) -> Annotated[Dict[str, float], "metrics"]:
    """Evaluate the assistant's performance with test questions.

    Args:
        config: Configuration for the RAG assistant
        test_questions: List of test questions to evaluate

    Returns:
        Dictionary containing evaluation metrics:
        - response_count: Number of non-empty responses
        - avg_response_length: Average length of responses
        - success_rate: Percentage of questions that got responses
    """
    try:
        # Load vector store
        logger.info(
            "Loading %s vector store from %s",
            config.vector_store.store_type.value,
            config.vector_store.store_path,
        )
        vector_store = VectorStoreFactory.load_store(
            config.vector_store,
            EMBEDDINGS,
        )
        if not vector_store:
            raise ValueError(
                f"Failed to load vector store from {config.vector_store.store_path}"
            )

        # Log test configuration and results
        log_metadata(
            metadata={
                "evaluation": {
                    "configuration": {
                        "num_questions": len(test_questions),
                        "questions": test_questions,
                        "llm_model": config.llm.model,
                        "temperature": config.llm.temperature,
                    },
                    "responses": {},  # Will be populated as we process responses
                    "summary": {
                        "total_questions": len(test_questions),
                    },
                }
            }
        )

        # Rebuild graph
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

        graph = graph_builder.compile()

        # Run test queries
        responses = []
        for i, question in enumerate(test_questions):
            try:
                response = graph.invoke({"question": question})["answer"]
                responses.append(response)
                logger.info("Question: %s", question)
                logger.info("Response: %s", response)

                # Update response metadata
                log_metadata(
                    metadata={
                        "evaluation": {
                            "responses": {
                                f"response_{i + 1}": {
                                    "question": question,
                                    "response": response,
                                    "response_length": len(response),
                                }
                            }
                        }
                    }
                )
            except Exception as e:
                logger.error(
                    "Failed to get response for question '%s': %s", question, e
                )
                responses.append("")

        # Calculate metrics
        non_empty_responses = [r for r in responses if r]
        metrics = {
            "response_count": len(non_empty_responses),
            "avg_response_length": (
                sum(len(r) for r in responses) / len(responses) if responses else 0
            ),
            "success_rate": len(non_empty_responses) / len(test_questions)
            if test_questions
            else 0,
        }

        # Log final summary
        log_metadata(
            metadata={
                "evaluation": {
                    "summary": {
                        "successful_responses": len(non_empty_responses),
                        "failed_responses": len(test_questions)
                        - len(non_empty_responses),
                        "metrics": metrics,
                    }
                }
            }
        )

        return metrics

    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise
