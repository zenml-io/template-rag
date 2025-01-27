from steps.create_assistant import create_assistant
from steps.evaluate_assistant import evaluate_assistant
from steps.ingest_and_embed import ingest_and_embed
from zenml import pipeline


@pipeline(enable_cache=False)
def create_assistant_pipeline():
    index = ingest_and_embed(data_path="data/")
    assistant = create_assistant(index)
    evaluate_assistant(assistant)
