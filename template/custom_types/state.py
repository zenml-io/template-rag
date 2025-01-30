"""Types related to the RAG assistant's state management."""

from typing import List

from langchain_core.documents import Document
from typing_extensions import TypedDict


class State(TypedDict):
    """Type definition for the RAG assistant's state.

    Attributes:
        question: The user's input question
        context: List of relevant documents retrieved for the question
        answer: The generated answer based on the context
    """

    question: str
    context: List[Document]
    answer: str
