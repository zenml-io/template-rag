"""Pipeline for creating and evaluating a RAG assistant."""

import os
from typing import List

from steps.create_assistant import create_assistant
from steps.evaluate_assistant import evaluate_assistant
from steps.ingest_and_embed import ingest_and_embed
from config.assistant import AssistantConfig
from config.constants import (
    DEFAULT_DOCS_PATH,
    DEFAULT_LLM_MODEL,
    DEFAULT_QA_TEMPLATE,
    DEFAULT_TEST_QUESTIONS,
)

from zenml import Model, pipeline
from zenml.config import DockerSettings

model = Model(
    name="ZenMLDocsAssistant",
    description="This is an assistant that answers questions about the ZenML documentation",
    license="MIT",
    audience="Developers",
    use_cases="Answer questions about the ZenML documentation",
    limitations="Limited to some parts of the documentation",
    trade_offs="No guarantees on the accuracy of the answers",
    tags=["llmops", "assistant", "rag"],
)


@pipeline(
    name="create_assistant_pipeline",
    enable_cache=False,
    model=model,
    settings={
        "docker": DockerSettings(
            requirements="requirements.txt",
            environment={
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            },
        ),
    },
)
def create_assistant_pipeline(
    assistant_config: AssistantConfig = AssistantConfig(),
    test_questions: List[str] = DEFAULT_TEST_QUESTIONS,
) -> None:
    """Create a RAG assistant pipeline.

    Args:
        assistant_config: Configuration for the RAG assistant
        test_questions: List of test questions to evaluate
    """
    # Ingest and embed documents
    vector_store = ingest_and_embed(config=assistant_config)

    # Create assistant
    graph = create_assistant(config=assistant_config, vector_store=vector_store)

    # Evaluate assistant
    evaluate_assistant(
        graph=graph,
        test_questions=test_questions,
        llm_config={
            "model": assistant_config.llm.model,
            "temperature": assistant_config.llm.temperature
        }
    )
