"""Tests for serialization utility functions."""

import io
from typing import Dict, Any
import pytest
import dill
from unittest.mock import MagicMock, mock_open, patch

from utils.serialization import save_vector_store, load_vector_store_data


def test_save_vector_store(mock_vector_store, mock_artifact_store):
    """Test saving vector store data."""
    store_path = "test/path/store.dill"

    # Create a mock file object
    mock_file = MagicMock(spec=io.BufferedWriter)
    mock_artifact_store.open.return_value.__enter__.return_value = mock_file

    # Call the function
    save_vector_store(mock_vector_store, store_path, mock_artifact_store)

    # Verify the artifact store was used correctly
    mock_artifact_store.open.assert_called_once_with(store_path, "wb")

    # Verify something was written to the file
    assert mock_file.write.called


def test_save_vector_store_data_format(mock_vector_store, mock_artifact_store):
    """Test the format of saved vector store data."""
    store_path = "test/path/store.dill"

    # Create a BytesIO to capture the written data
    written_data = io.BytesIO()
    mock_artifact_store.open.return_value.__enter__.return_value = written_data

    # Call the function
    save_vector_store(mock_vector_store, store_path, mock_artifact_store)

    # Get the written data and load it
    written_data.seek(0)
    saved_data = dill.loads(written_data.getvalue())

    # Verify the structure of saved data
    assert isinstance(saved_data, dict)
    assert "texts" in saved_data
    assert "metadatas" in saved_data
    assert "embeddings" in saved_data
    assert isinstance(saved_data["texts"], list)
    assert isinstance(saved_data["metadatas"], list)
    assert isinstance(saved_data["embeddings"], list)


def test_load_vector_store_data(mock_artifact_store):
    """Test loading vector store data."""
    store_path = "test/path/store.dill"

    # Create test data
    test_data = {
        "texts": ["doc1", "doc2"],
        "metadatas": [{"source": "test1.txt"}, {"source": "test2.txt"}],
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],
    }

    # Create a BytesIO with the test data
    data_bytes = dill.dumps(test_data)
    mock_file = io.BytesIO(data_bytes)
    mock_artifact_store.open.return_value.__enter__.return_value = mock_file

    # Load the data
    loaded_data = load_vector_store_data(store_path, mock_artifact_store)

    # Verify the loaded data
    assert loaded_data == test_data
    assert isinstance(loaded_data["texts"], list)
    assert isinstance(loaded_data["metadatas"], list)
    assert isinstance(loaded_data["embeddings"], list)
    assert len(loaded_data["texts"]) == 2
    assert len(loaded_data["metadatas"]) == 2
    assert len(loaded_data["embeddings"]) == 2


def test_load_vector_store_data_file_not_found(mock_artifact_store):
    """Test loading vector store data with file not found."""
    store_path = "nonexistent/path/store.dill"

    # Mock the artifact store to raise FileNotFoundError
    mock_artifact_store.open.side_effect = FileNotFoundError

    # Verify that the function raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_vector_store_data(store_path, mock_artifact_store)


def test_save_vector_store_permission_error(mock_vector_store, mock_artifact_store):
    """Test saving vector store data with permission error."""
    store_path = "test/path/store.dill"

    # Mock the artifact store to raise PermissionError
    mock_artifact_store.open.side_effect = PermissionError

    # Verify that the function raises PermissionError
    with pytest.raises(PermissionError):
        save_vector_store(mock_vector_store, store_path, mock_artifact_store)
