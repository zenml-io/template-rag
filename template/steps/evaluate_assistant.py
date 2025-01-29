"""Step to evaluate the RAG assistant's performance."""

import logging
from typing import Annotated, Dict, List

from langchain_community.vectorstores import FAISS
from langgraph.graph import START, StateGraph
from zenml import step

from config.models import EMBEDDINGS
from config.store import AssistantConfig
from custom_types.state import State
from utils.graph import generate
from utils.vector_store import create_retrieve

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
    # Reconstruct vector store
    vector_store = FAISS.load_local(
        config.vector_store.store_path,
        EMBEDDINGS,
        allow_dangerous_deserialization=True,
    )

    # Rebuild graph
    retrieve = create_retrieve(vector_store)
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    # Run test queries
    responses = []
    for question in test_questions:
        try:
            response = graph.invoke({"question": question})["answer"]
            responses.append(response)
            logger.info("Question: %s", question)
            logger.info("Response: %s", response)
        except Exception as e:
            logger.error("Failed to get response for question '%s': %s", question, e)
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

    return metrics
