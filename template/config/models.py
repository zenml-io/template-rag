"""Configuration for language models and embeddings."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Model configurations
LLM = ChatOpenAI(model="gpt-4o-mini")
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")
