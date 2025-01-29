"""Tests for configuration classes."""

import pytest
from pydantic import ValidationError

from config.store import AssistantConfig, LLMConfig, VectorStoreConfig


def test_vector_store_config_validation():
    """Test VectorStoreConfig validation."""
    # Test valid config
    config = VectorStoreConfig(
        store_type="faiss",
        store_path="data/test",
        embedding_model="text-embedding-3-small",
    )
    assert config.store_type == "faiss"
    assert config.store_path == "data/test"
    assert config.embedding_model == "text-embedding-3-small"

    # Test invalid store type
    with pytest.raises(ValidationError):
        VectorStoreConfig(
            store_type="invalid",
            store_path="data/test",
            embedding_model="text-embedding-3-small",
        )

    # Test invalid embedding model
    with pytest.raises(ValidationError):
        VectorStoreConfig(
            store_type="faiss",
            store_path="data/test",
            embedding_model="invalid",
        )

    # Test empty store path
    with pytest.raises(ValueError):
        VectorStoreConfig(
            store_type="faiss",
            store_path="",
            embedding_model="text-embedding-3-small",
        )

    # Test invalid chunk settings
    with pytest.raises(ValueError):
        VectorStoreConfig(
            store_type="faiss",
            store_path="data/test",
            embedding_model="text-embedding-3-small",
            chunk_size=100,
            chunk_overlap=200,  # overlap > size
        )


def test_llm_config_validation():
    """Test LLMConfig validation."""
    # Test valid config
    config = LLMConfig(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )
    assert config.model == "gpt-3.5-turbo"
    assert config.temperature == 0.7
    assert config.streaming is False  # default

    # Test invalid model
    with pytest.raises(ValidationError):
        LLMConfig(
            model="invalid",
            temperature=0.7,
        )

    # Test temperature bounds
    with pytest.raises(ValidationError):
        LLMConfig(
            model="gpt-3.5-turbo",
            temperature=-0.1,  # too low
        )
    with pytest.raises(ValidationError):
        LLMConfig(
            model="gpt-3.5-turbo",
            temperature=2.1,  # too high
        )

    # Test max_tokens validation
    with pytest.raises(ValidationError):
        LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=-1,  # must be positive
        )


def test_assistant_config_validation():
    """Test AssistantConfig validation."""
    # Create valid sub-configs
    vector_store = VectorStoreConfig(
        store_type="faiss",
        store_path="data/test",
        embedding_model="text-embedding-3-small",
    )
    llm = LLMConfig(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )

    # Test valid config
    config = AssistantConfig(
        vector_store=vector_store,
        llm=llm,
        prompt_templates={
            "qa": "Context: {context}\nQuestion: {question}\nAnswer:",
        },
    )
    assert config.vector_store == vector_store
    assert config.llm == llm

    # Test missing required placeholders in template
    with pytest.raises(ValueError):
        AssistantConfig(
            vector_store=vector_store,
            llm=llm,
            prompt_templates={
                "qa": "Invalid template without placeholders",
            },
        )

    # Test missing context placeholder
    with pytest.raises(ValueError):
        AssistantConfig(
            vector_store=vector_store,
            llm=llm,
            prompt_templates={
                "qa": "Question: {question}\nAnswer:",  # missing {context}
            },
        )

    # Test missing question placeholder
    with pytest.raises(ValueError):
        AssistantConfig(
            vector_store=vector_store,
            llm=llm,
            prompt_templates={
                "qa": "Context: {context}\nAnswer:",  # missing {question}
            },
        )
