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
        task="openai/gpt-5-mini",
        teacher="openai/gpt-5",
        embedding="openai/text-embedding-3-small",
    ),
    corpus=dp.Document(
        name="ECB Supervisory Manual",
        pages=dp.utils.load_pdf(path="documents/ecb_supman.pdf"),
    ),
    dataset=[
        dp.Examples(qa_pairs=dp.utils.load_json(path="examples/1_out_of_scope.json")),
        dp.Examples(qa_pairs=dp.utils.load_json(path="examples/2_unanswerable.json")),
        dp.Examples(qa_pairs=dp.utils.load_json(path="examples/3_sin_ecb_supman.json")),
        dp.Examples(qa_pairs=dp.utils.load_json(path="examples/4_mul_ecb_supman.json")),
    ],
    metrics={
        "composite": dp.metrics.Composite(judge_model="openai/gpt-5"),
    },
)

if __name__ == "__main__":
    expert_agent.ask(question="What does SSM stand for?")
    expert_agent.evaluate(metric="composite")
    expert_agent.optimize(metric="composite", effort="light")
