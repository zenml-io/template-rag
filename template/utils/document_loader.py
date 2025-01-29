"""Utility functions for loading and processing documents."""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_documents(
    docs_path: str, chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Load and process documents from a directory.

    Args:
        docs_path: Path to the directory containing documents
        chunk_size: Size of text chunks for splitting documents
        chunk_overlap: Overlap between text chunks

    Returns:
        List of processed document chunks

    Raises:
        FileNotFoundError: If no documents are found in the specified path
        ValueError: If document processing fails
    """
    try:
        # Load markdown files from directory
        logger.info("Loading markdown files from %s", docs_path)
        loader = DirectoryLoader(
            docs_path,
            glob="**/*.md",  # Recursively find all markdown files
        )
        docs = loader.load()
        if not docs:
            raise FileNotFoundError(f"No markdown files found in {docs_path}")

        # Split documents into chunks
        logger.info("Splitting %d documents into chunks", len(docs))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(docs)
        logger.info("Created %d text chunks", len(chunks))

        return chunks

    except Exception as e:
        logger.error("Failed to load documents: %s", e)
        raise
