import os

import chainlit as cl
from traceloop.sdk import Traceloop
from zenml.client import Client

TRACELOOP_API_KEY = os.getenv("TRACELOOP_API_KEY")


@cl.on_chat_start
async def init() -> None:
    """Initialize the chat session by loading the query engine."""
    try:
        Traceloop.init(api_key=TRACELOOP_API_KEY)
        # Load the query engine and store it in the user session
        artifact = Client().get_artifact_version("7040849f-53c3-44d6-95e0-31949d28a773")
        query_engine = artifact.load()
        cl.user_session.set("query_engine", query_engine)

        # Send initial message
        await cl.Message(
            content="👋 Hello! I'm ready to answer your questions about ZenML."
        ).send()
    except Exception as e:
        await cl.Message(content=f"Error initializing chat: {str(e)}").send()


@cl.on_message
async def main(message: cl.Message) -> None:
    """Handle incoming user messages.

    Args:
        message: The incoming message from the user
    """
    try:
        # Get query engine from user session
        query_engine = cl.user_session.get("query_engine")

        # Get response from query engine
        response = query_engine.query(message.content)

        # Send response back to user
        await cl.Message(content=str(response)).send()
    except Exception as e:
        await cl.Message(content=f"Error processing your question: {str(e)}").send()
