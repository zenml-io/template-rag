"""Step to create a RAG assistant using a vector store."""

from typing import Annotated, List

from config.models import LLM
from langchain import hub
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from materializers.state_graph_materializer import StateGraphMaterializer
from typing_extensions import TypedDict
from zenml import step


@step(output_materializers=StateGraphMaterializer)
def create_assistant(
    vector_store: InMemoryVectorStore,
) -> Annotated[CompiledStateGraph, "assistant"]:
    """Create a RAG assistant using a vector store.

    Args:
        vector_store: Vector store containing document embeddings

    Returns:
        A compiled state graph that can be used as a RAG assistant
    """
    # Define prompt for question-answering
    prompt = hub.pull("rlm/rag-prompt")

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        response = LLM.invoke(messages)
        return {"answer": response.content}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph
