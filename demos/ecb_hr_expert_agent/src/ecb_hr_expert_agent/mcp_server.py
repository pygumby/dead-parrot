"""Expert agent MCP server."""

import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import dotenv
import temporalio.client
from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan

from .constants import EXPERT_AGENT_DESCRIPTION, EXPERT_AGENT_NAME

dotenv.load_dotenv()


@lifespan
async def app_lifespan(
    mcp: FastMCP,
) -> AsyncGenerator[dict[str, temporalio.client.Client], None]:
    """Lifespan context manager."""
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost")
    temporal_port = os.getenv("TEMPORAL_PORT", "7233")
    temporal_url = f"{temporal_host}:{temporal_port}"
    temporal_client = await temporalio.client.Client.connect(temporal_url)
    yield {"temporal_client": temporal_client}


mcp = FastMCP(
    name=EXPERT_AGENT_NAME,
    lifespan=app_lifespan,
)


@mcp.tool(name=EXPERT_AGENT_NAME, description=EXPERT_AGENT_DESCRIPTION)
async def ask(ctx: Context, question: str) -> dict[str, Any]:
    """Ask a question to the expert agent."""
    temporal_client: temporalio.client.Client = ctx.lifespan_context["temporal_client"]
    response: dict[str, Any] = await temporal_client.execute_workflow(
        workflow="EcbHrExpertAgentWorkflow",
        arg=question,
        id=f"{EXPERT_AGENT_NAME}-mcp-{uuid.uuid4().hex[:8]}",
        task_queue=f"{EXPERT_AGENT_NAME}-queue",
    )
    return response


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "localhost")
    port = int(os.getenv("MCP_PORT", "9002"))
    mcp.run(transport="http", host=host, port=port)
