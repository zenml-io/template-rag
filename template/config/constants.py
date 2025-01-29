"""Constants used throughout the template."""

from enum import Enum
from typing import Literal

# Vector store settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_SEARCH_RESULTS = 4


# Supported model types
class VectorStoreType(str, Enum):
    """Supported vector store types."""

    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"


SUPPORTED_EMBEDDING_MODELS = Literal["text-embedding-3-small", "text-embedding-3-large"]
SUPPORTED_LLM_MODELS = Literal["gpt-3.5-turbo", "gpt-4"]

# Default paths
DEFAULT_VECTOR_STORE_PATH = "data/vector_store"
DEFAULT_DOCS_PATH = "data/docs"

# Default model settings
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.7

# Default prompt templates
DEFAULT_QA_TEMPLATE = """Use the following context to answer the question:

Context: {context}

Question: {question}

Answer:"""

# Validation settings
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 8000
MIN_CHUNK_OVERLAP = 0
MAX_CHUNK_OVERLAP = 1000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
