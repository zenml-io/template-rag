"""Configuration classes for vector stores and assistants."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from config.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_QA_TEMPLATE,
    DEFAULT_TEMPERATURE,
    MAX_CHUNK_OVERLAP,
    MAX_CHUNK_SIZE,
    MAX_TEMPERATURE,
    MIN_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    MIN_TEMPERATURE,
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_LLM_MODELS,
    SUPPORTED_VECTOR_STORES,
)


class VectorStoreConfig(BaseModel):
    """Configuration for vector store access."""

    store_type: SUPPORTED_VECTOR_STORES = Field(
        description="Type of vector store to use"
    )
    store_path: str = Field(
        description="Local path or remote URI for vector store data"
    )
    embedding_model: SUPPORTED_EMBEDDING_MODELS = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        description="Name of the embedding model to use",
    )
    documents_path: Optional[str] = Field(
        default=None, description="Path to the original documents"
    )
    chunk_size: Optional[int] = Field(
        default=1000,
        description="Size of text chunks for splitting documents",
        gt=MIN_CHUNK_SIZE,
        lt=MAX_CHUNK_SIZE,
    )
    chunk_overlap: Optional[int] = Field(
        default=200,
        description="Overlap between text chunks",
        ge=MIN_CHUNK_OVERLAP,
        le=MAX_CHUNK_OVERLAP,
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata for the vector store"
    )

    @field_validator("store_path")
    def validate_store_path(cls, v: str) -> str:
        """Validate store path exists or is a valid URI."""
        # In production, we would:
        # 1. Check if local path exists
        # 2. Validate remote URIs
        # 3. Check permissions
        if not v:
            raise ValueError("store_path cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "VectorStoreConfig":
        """Validate chunk size and overlap are compatible."""
        if (
            self.chunk_size is not None
            and self.chunk_overlap is not None
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class LLMConfig(BaseModel):
    """Configuration for the language model."""

    model: SUPPORTED_LLM_MODELS = Field(
        default=DEFAULT_LLM_MODEL, description="Name of the LLM model to use"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="Temperature for LLM generation",
        ge=MIN_TEMPERATURE,
        le=MAX_TEMPERATURE,
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to generate", gt=0
    )
    streaming: bool = Field(default=False, description="Whether to stream the response")


class AssistantConfig(BaseModel):
    """Configuration for the RAG assistant."""

    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store"
    )
    llm: LLMConfig = Field(description="Configuration for the language model")
    prompt_templates: Dict[str, str] = Field(
        default_factory=lambda: {"qa": DEFAULT_QA_TEMPLATE},
        description="Templates for different prompt types",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata for the assistant"
    )

    @field_validator("prompt_templates")
    def validate_prompt_templates(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate prompt templates contain required placeholders."""
        required_placeholders = ["{context}", "{question}"]
        for template_name, template in v.items():
            missing = [p for p in required_placeholders if p not in template]
            if missing:
                raise ValueError(
                    f"Template '{template_name}' missing placeholders: {missing}"
                )
        return v
