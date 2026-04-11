"""AI Assistant REST API."""

import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from temporalio.client import Client

from .constants import AI_ASSISTANT_DESCRIPTION, AI_ASSISTANT_NAME
from .temporal_workflow import EcbHrAiAssistantWorkflow

dotenv.load_dotenv()


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager."""
    temporal_service_url = os.getenv("TEMPORAL_SERVICE_URL", "localhost:7233")
    app.state.temporal_client = await Client.connect(temporal_service_url)
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
        EcbHrAiAssistantWorkflow.run,
        body.question,
        id=f"{AI_ASSISTANT_NAME}-{uuid.uuid4().hex[:8]}",
        task_queue=f"{AI_ASSISTANT_NAME}-queue",
    )
    return response
