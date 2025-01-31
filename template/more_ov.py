from typing import Annotated, Any, Type
import bs4

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub

# Custom Materializer
from zenml import Model, get_step_context, pipeline, step
from zenml.materializers.base_materializer import BaseMaterializer
import json
import dill


llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)


# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

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
    ASSOCIATED_TYPES = (CompiledStateGraph,)

    file_name = "state_graph.pkl"

    def load(self, data_type: Type[CompiledStateGraph]) -> CompiledStateGraph:
        with open(self.uri + "/" + self.file_name, "rb") as f:
            loaded_data = dill.load(f)
        return loaded_data

    def save(self, data: CompiledStateGraph) -> None:
        serialized_data = dill.dumps(data)
        with open(self.uri + "/" + self.file_name, "wb") as f:
            f.write(serialized_data)


@step(output_materializers=StateGraphMaterializer)
def rag_logic() -> Annotated[CompiledStateGraph, "rag_graph"]:
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph


# @step
# def evaluate_rag(cg: CompiledStateGraph) -> str:
#     """Evaluate the RAG assistant with a test question."""
#     prompt = {"question": "What are ZenML service connectors?"}
#     response = cg.invoke(prompt)
#     print(response["answer"])
#     return response["answer"]


@pipeline(model=Model(name="rag_langgraph", version="dev"))
def rag_langgraph_build_evaluate_pipeline():
    graph = rag_logic()
    # evaluate_rag(graph)


if __name__ == "__main__":
    rag_langgraph_build_evaluate_pipeline()
