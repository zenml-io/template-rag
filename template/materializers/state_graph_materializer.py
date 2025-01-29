import os
from typing import Any, ClassVar, List, Optional, Tuple, Type

import dill
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict
from zenml.artifact_stores.base_artifact_store import BaseArtifactStore
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def create_retrieve(vector_store):
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    return retrieve


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = hub.pull("rlm/rag-prompt")
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(messages)
    return {"answer": response.content}


class StateGraphMaterializer(BaseMaterializer):
    """Materializer to handle CompiledStateGraph objects."""

    ASSOCIATED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (CompiledStateGraph,)
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.DATA

    def __init__(self, uri: str, artifact_store: Optional[BaseArtifactStore] = None):
        """Initialize the materializer.

        Args:
            uri: The URI where the artifact data is stored.
            artifact_store: The artifact store where the artifact data is stored.
        """
        super().__init__(uri, artifact_store)
        self.store_path = os.path.join(self.uri, "state_graph.dill")

    def load(self, data_type: Type[Any]) -> CompiledStateGraph:
        """Read from artifact store and reconstruct the graph.

        Args:
            data_type: The type of the data to read.

        Returns:
            CompiledStateGraph: The reconstructed state graph.
        """
        with self.artifact_store.open(self.store_path, "rb") as f:
            data = dill.load(f)

        # Reconstruct the vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        texts = data["texts"]
        metadatas = data["metadatas"]

        # Create a FAISS vector store with the documents
        vector_store = FAISS.from_texts(
            texts=texts, embedding=embeddings, metadatas=metadatas
        )

        # Build the graph
        retrieve = create_retrieve(vector_store)
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        return graph

    def save(self, graph: CompiledStateGraph) -> None:
        """Write essential components to artifact store.

        Args:
            graph: The state graph to write.
        """
        # Extract the vector store from the graph's state
        vector_store = None

        # Helper function to recursively search for vector store in object attributes
        def find_vector_store(obj, visited=None):
            if visited is None:
                visited = set()

            # Avoid circular references
            obj_id = id(obj)
            if obj_id in visited:
                return None
            visited.add(obj_id)

            # Check if this object has similarity_search method
            if hasattr(obj, "similarity_search"):
                return obj

            # If it's a function, check its closure
            if isinstance(obj, type(lambda: None)):
                if hasattr(obj, "__closure__") and obj.__closure__:
                    for cell in obj.__closure__:
                        result = find_vector_store(cell.cell_contents, visited)
                        if result:
                            return result

            # If it's a dictionary, check its values
            elif isinstance(obj, dict):
                for value in obj.values():
                    result = find_vector_store(value, visited)
                    if result:
                        return result

            # If it has __dict__, check all its attributes
            elif hasattr(obj, "__dict__"):
                for value in obj.__dict__.values():
                    result = find_vector_store(value, visited)
                    if result:
                        return result

            # If it's a list or tuple, check its elements
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    result = find_vector_store(item, visited)
                    if result:
                        return result

            return None

        # Search for vector store in graph nodes and edges
        vector_store = find_vector_store(graph)

        if vector_store is None:
            raise ValueError(
                "Could not extract vector store from graph. This is required for serialization."
            )

        # Extract just the documents and their metadata
        documents = []
        for doc in vector_store.similarity_search("", k=1000):  # Get all documents
            documents.append(doc)

        # Store only the essential data
        data = {
            "texts": [doc.page_content for doc in documents],
            "metadatas": [doc.metadata for doc in documents],
        }

        with self.artifact_store.open(self.store_path, "wb") as f:
            dill.dump(data, f)
