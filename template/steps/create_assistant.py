from typing import Annotated

from langchain import hub
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from materializers.state_graph_materializer import StateGraphMaterializer
from typing_extensions import List, TypedDict
from zenml import step

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def create_retrieve(vector_store: InMemoryVectorStore):
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    return retrieve


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


@step(output_materializers=StateGraphMaterializer)
def create_assistant(
    vector_store: InMemoryVectorStore,
) -> Annotated[CompiledStateGraph, "assistant"]:
    retrieve = create_retrieve(vector_store)
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph
