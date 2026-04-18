"""Triage agent."""

import getpass
import os

import dotenv

import dead_parrot as dp

from .constants import TRIAGE_AGENT_NAME

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter OpenAI API key: ")

triage_agent = dp.TriageAgent(
    name=TRIAGE_AGENT_NAME,
    task_model="gpt-5-mini",
    expert_agent_clients=[
        dp.ExpertAgentClient(scheme="http", host="localhost", port=8001),
        dp.ExpertAgentClient(scheme="http", host="localhost", port=8002),
    ],
    dataset=dp.Examples(
        qa_pairs=dp.utils.load_json(path="examples/ecb_supervisory_manual.json"),
    ),
    metrics={
        "recall": dp.metrics.SimpleRecall(judge_model="gpt-5"),
        "sources": dp.metrics.SimpleSourcesCoverage(judge_model="gpt-5"),
    },
)

if __name__ == "__main__":
    triage_agent.ask(
        question="What does SSM stand for? How long is the probationary period?"
    )
    triage_agent.evaluate(metric="recall")
