"""AI Agent Temporal Worker."""

import asyncio
import os

import dotenv
from dspy_temporal import DSPyPlugin
from dspy_temporal.sandbox import (
    get_default_sandbox_restrictions,
    get_default_sandbox_runner,
)
from temporalio.client import Client
from temporalio.worker import Worker

from .constants import AI_AGENT_NAME, PACKAGE_NAME
from .temporal_workflow import EcbAiAgentWorkflow, react

dotenv.load_dotenv()


async def run_temporal_worker() -> None:
    """Run the Temporal worker."""
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost")
    temporal_port = os.getenv("TEMPORAL_PORT", "7233")
    temporal_url = f"{temporal_host}:{temporal_port}"
    temporal_client: Client = await Client.connect(temporal_url)
    temporal_worker: Worker = Worker(
        temporal_client,
        task_queue=f"{AI_AGENT_NAME}-queue",
        workflows=[EcbAiAgentWorkflow],
        plugins=[DSPyPlugin(react)],
        workflow_runner=get_default_sandbox_runner(
            get_default_sandbox_restrictions().with_passthrough_modules(
                "dead_parrot",
                PACKAGE_NAME,
            ),
        ),
    )
    await temporal_worker.run()


if __name__ == "__main__":
    asyncio.run(run_temporal_worker())
