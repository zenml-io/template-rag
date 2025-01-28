import os
from steps.create_assistant import create_assistant
from steps.evaluate_assistant import evaluate_assistant
from steps.ingest_and_embed import ingest_and_embed
from zenml import pipeline
from zenml import Model
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
    enable_cache=False,
    model=model,
    settings={
        "docker": DockerSettings(
            requirements="requirements.txt",
            environment={
                "TRACELOOP_API_KEY": os.getenv("TRACELOOP_API_KEY"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            },
        ),
    },
)
def create_assistant_pipeline():
    index = ingest_and_embed(data_path="data/")
    assistant = create_assistant(index)
    evaluate_assistant(assistant)
