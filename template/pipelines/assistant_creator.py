import os

from steps.create_assistant import create_assistant
from steps.ingest_and_embed import ingest_and_embed
from zenml import Model, pipeline
from zenml.config import DockerSettings

model = Model(
    name="ZenMLDocsAssistant",
    description="This is an assistant that can answer questions about the ZenML documentation",
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
def create_assistant_pipeline():
    """Create a RAG assistant pipeline.

    This pipeline:
    1. Ingests documents and creates embeddings
    2. Creates a RAG assistant using the embeddings
    3. Evaluates the assistant's performance
    """
    # First create the vector store with document embeddings
    vector_store = ingest_and_embed(data_path="data/")

    # # Create the assistant using the vector store
    assistant = create_assistant(vector_store=vector_store)

    # # Evaluate the assistant
    # evaluate_assistant(assistant=assistant)
