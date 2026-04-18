"""Expert agent."""

import getpass
import os

import dotenv

import dead_parrot as dp

from .constants import EXPERT_AGENT_NAME

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter OpenAI API key: ")

expert_agent = dp.ExpertAgent(
    name=EXPERT_AGENT_NAME,
    models=dp.Models(
        task="gpt-5-mini",
        teacher="gpt-5",
        embedding="text-embedding-3-small",
    ),
    corpus=[
        dp.Document(
            name="European Central Bank Staff Rules",
            pages=dp.utils.load_pdf(path="documents/ecb_rules.pdf"),
        ),
        dp.Document(
            name="European Central Bank Conditions of Employment",
            pages=dp.utils.load_pdf(path="documents/ecb_conditions.pdf"),
        ),
    ],
    dataset=[
        dp.Examples(qa_pairs=dp.utils.load_json(path="examples/ecb_rules.json")),
        dp.Examples(qa_pairs=dp.utils.load_json(path="examples/ecb_conditions.json")),
    ],
    metrics={
        "recall": dp.metrics.SimpleRecall(judge_model="gpt-5"),
        "sources": dp.metrics.SimpleSourcesCoverage(judge_model="gpt-5"),
    },
)

if __name__ == "__main__":
    expert_agent.ask(question="How long is the probationary period?")
    expert_agent.evaluate(metric="recall")
    expert_agent.optimize(metric="recall", effort="light")
