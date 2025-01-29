"""Tests for configuration classes."""

import pytest
from pydantic import ValidationError

from config.constants import VectorStoreType
from config.store import AssistantConfig, LLMConfig, VectorStoreConfig


def test_vector_store_config_validation():
    """Test VectorStoreConfig validation."""
    # Test valid config
    config = VectorStoreConfig(
        store_type=VectorStoreType.FAISS,
        store_path="data/test",
        embedding_model="text-embedding-3-small",
    )
    assert config.store_type == VectorStoreType.FAISS
    assert config.store_path == "data/test"
    assert config.embedding_model == "text-embedding-3-small"

    # Test invalid store type
    with pytest.raises(ValidationError):
        VectorStoreConfig(
            store_type="invalid",  # type: ignore
            store_path="data/test",
            embedding_model="text-embedding-3-small",
        )

    # Test invalid embedding model
    with pytest.raises(ValidationError):
        VectorStoreConfig(
            store_type=VectorStoreType.FAISS,
            store_path="data/test",
            embedding_model="invalid",
        )

    # Test empty store path
    with pytest.raises(ValueError):
        VectorStoreConfig(
            store_type=VectorStoreType.FAISS,
            store_path="",
            embedding_model="text-embedding-3-small",
        )

    # Test invalid chunk settings
    with pytest.raises(ValueError):
        VectorStoreConfig(
            store_type=VectorStoreType.FAISS,
            store_path="data/test",
            embedding_model="text-embedding-3-small",
            chunk_size=100,
            chunk_overlap=200,  # overlap > size
        )


def test_llm_config_validation():
    """Test LLMConfig validation."""
    # Test valid config
    config = LLMConfig()  # Test defaults
    assert config.model == "gpt-3.5-turbo"
    assert config.temperature == 0.7
    assert not config.streaming

    # Test custom values
    config = LLMConfig(
        model="gpt-4",
        temperature=0.5,
        max_tokens=100,
        streaming=True,
    )
    assert config.model == "gpt-4"
    assert config.temperature == 0.5
    assert config.max_tokens == 100
    assert config.streaming

    # Test invalid model
    with pytest.raises(ValidationError):
        LLMConfig(model="invalid")

    # Test invalid temperature
    with pytest.raises(ValidationError):
        LLMConfig(temperature=2.5)

    # Test invalid max_tokens
    with pytest.raises(ValidationError):
        LLMConfig(max_tokens=0)


def test_assistant_config_validation():
    """Test AssistantConfig validation."""
    # Test valid config
    vector_store = VectorStoreConfig(
        store_type=VectorStoreType.FAISS,
        store_path="data/test",
        embedding_model="text-embedding-3-small",
    )
    llm = LLMConfig()
    config = AssistantConfig(vector_store=vector_store, llm=llm)
    assert config.vector_store == vector_store
    assert config.llm == llm
    assert "qa" in config.prompt_templates

    # Test invalid prompt template
    with pytest.raises(ValidationError):
        AssistantConfig(
            vector_store=vector_store,
            llm=llm,
            prompt_templates={"qa": "Invalid template without placeholders"},
        )

    # Test valid custom prompt template
    config = AssistantConfig(
        vector_store=vector_store,
        llm=llm,
        prompt_templates={
            "qa": "Context: {context}\nQ: {question}\nA:",
            "custom": "Using this info: {context}\nAnswer: {question}",
        },
    )
    assert len(config.prompt_templates) == 2
