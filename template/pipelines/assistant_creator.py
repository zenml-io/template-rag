import os

from steps.create_assistant import create_assistant
from steps.evaluate_assistant import evaluate_assistant
from steps.ingest_and_embed import ingest_and_embed
from zenml import Model, pipeline
from zenml.config import DockerSettings

from config.constants import (
    DEFAULT_DOCS_PATH,
    DEFAULT_LLM_MODEL,
    DEFAULT_QA_TEMPLATE,
)

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
    enable_cache=True,
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
    docs_path: str = DEFAULT_DOCS_PATH,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.7,
    qa_template: str = DEFAULT_QA_TEMPLATE,
):
    """Create a RAG assistant pipeline.

    Args:
        docs_path: Path to the documents to ingest
        model: Name of the LLM model to use
        temperature: Temperature for LLM generation
        qa_template: Template for QA prompts
    """
    # First create the vector store with document embeddings
    vector_store_config = ingest_and_embed(docs_path=docs_path)

    # Create the assistant using the vector store
    assistant_config = create_assistant(
        vector_store_config=vector_store_config,
        model=model,
        temperature=temperature,
        qa_template=qa_template,
    )

    # Evaluate the assistant
    evaluate_assistant(config=assistant_config)
