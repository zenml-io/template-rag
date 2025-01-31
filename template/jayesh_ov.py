import os
from typing import Annotated, Type

import dill
from config.constants import CHUNK_OVERLAP, CHUNK_SIZE
from langchain import hub
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import List, TypedDict
from zenml import Model, pipeline, step
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)


loader = DirectoryLoader(
    "data/",
    glob="**/*.md",
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
all_splits = text_splitter.split_documents(docs)

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


class StateGraphMaterializer(BaseMaterializer):
    """Materializer to handle CompiledStateGraph objects."""

    ASSOCIATED_TYPES = (CompiledStateGraph,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    file_name = "state_graph.pkl"

    def load(self, data_type: Type[CompiledStateGraph]) -> CompiledStateGraph:
        """Read from artifact store."""
        with self.artifact_store.open(
            os.path.join(self.uri, self.file_name), "rb"
        ) as f:
            loaded_data = dill.load(f)
        return loaded_data

    def save(self, data: CompiledStateGraph) -> None:
        """Write to artifact store."""
        serialized_data = dill.dumps(data)
        with self.artifact_store.open(
            os.path.join(self.uri, self.file_name), "wb"
        ) as f:
            f.write(serialized_data)


@step(output_materializers=StateGraphMaterializer)
def create_assistant() -> Annotated[CompiledStateGraph, "rag_graph"]:
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    print(graph)
    return graph


@step
def evaluate_rag(cg: CompiledStateGraph) -> str:
    """Evaluate the RAG assistant with a test question."""
    prompt = {"question": "What are ZenML service connectors?"}
    response = cg.invoke(prompt)
    print(response["answer"])
    return response["answer"]


@pipeline(model=Model(name="langgraph_assistant", version="dev"), enable_cache=False)
def rag_langgraph_assistant_builder():
    assistant = create_assistant()
    evaluate_rag(assistant)


if __name__ == "__main__":
    rag_langgraph_assistant_builder()
