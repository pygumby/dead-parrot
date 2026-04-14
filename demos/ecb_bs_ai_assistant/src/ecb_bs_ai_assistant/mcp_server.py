"""AI Assistant MCP Server."""

import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import temporalio.client
from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan

from .constants import AI_ASSISTANT_DESCRIPTION, AI_ASSISTANT_NAME


@lifespan
async def app_lifespan(
    mcp: FastMCP,
) -> AsyncGenerator[dict[str, temporalio.client.Client], None]:
    """Lifespan context manager."""
    temporal_service_url = os.getenv("TEMPORAL_SERVICE_URL", "localhost:7233")
    temporal_client = await temporalio.client.Client.connect(temporal_service_url)
    yield {"temporal_client": temporal_client}


mcp = FastMCP(
    name=AI_ASSISTANT_NAME,
    lifespan=app_lifespan,
)


@mcp.tool(name=AI_ASSISTANT_NAME, description=AI_ASSISTANT_DESCRIPTION)
async def ask(ctx: Context, question: str) -> dict[str, Any]:
    """Ask a question to the AI assistant."""
    temporal_client: temporalio.client.Client = ctx.lifespan_context["temporal_client"]
    response: dict[str, Any] = await temporal_client.execute_workflow(
        workflow="EcbBsAiAssistantWorkflow",
        arg=question,
        id=f"{AI_ASSISTANT_NAME}-{uuid.uuid4().hex[:8]}",
        task_queue=f"{AI_ASSISTANT_NAME}-queue",
    )
    return response


if __name__ == "__main__":
    mcp.run(transport="http", host="localhost", port=9001)
