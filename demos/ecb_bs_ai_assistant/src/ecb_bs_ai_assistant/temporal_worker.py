"""AI Assistant Temporal Worker."""

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

from .constants import AI_ASSISTANT_NAME, PACKAGE_NAME
from .temporal_workflow import EcbBsAiAssistantWorkflow, rag, retriever

dotenv.load_dotenv()


async def run_worker() -> None:
    """Run the Temporal worker."""
    url: str = os.getenv("TEMPORAL_SERVICE_URL", "localhost:7233")
    client: Client = await Client.connect(url)
    worker: Worker = Worker(
        client,
        task_queue=f"{AI_ASSISTANT_NAME}-queue",
        workflows=[EcbBsAiAssistantWorkflow],
        plugins=[DSPyPlugin(rag, retriever)],
        workflow_runner=get_default_sandbox_runner(
            get_default_sandbox_restrictions().with_passthrough_modules(
                "dead_parrot",
                PACKAGE_NAME,
            ),
        ),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())
