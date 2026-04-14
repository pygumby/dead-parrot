"""AI Assistant REST Server."""

import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import dotenv
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from temporalio.client import Client

from .constants import AI_ASSISTANT_DESCRIPTION, AI_ASSISTANT_NAME

dotenv.load_dotenv()


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager."""
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost")
    temporal_port = os.getenv("TEMPORAL_PORT", "7233")
    temporal_url = f"{temporal_host}:{temporal_port}"
    app.state.temporal_client = await Client.connect(temporal_url)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/card")
async def card() -> dict[str, str]:
    """Get the AI assistant's card."""
    return {"description": AI_ASSISTANT_DESCRIPTION}


@app.post("/ask")
async def ask(body: AskRequest, request: Request) -> dict[str, Any]:
    """Ask a question to the AI assistant."""
    response: dict[str, Any] = await request.app.state.temporal_client.execute_workflow(
        workflow="EcbBsAiAssistantWorkflow",
        arg=body.question,
        id=f"{AI_ASSISTANT_NAME}-{uuid.uuid4().hex[:8]}",
        task_queue=f"{AI_ASSISTANT_NAME}-queue",
    )
    return response


if __name__ == "__main__":
    host = os.getenv("REST_HOST", "localhost")
    port = int(os.getenv("REST_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
