"""AI Assistant."""

import getpass
import os

import dotenv

import dead_parrot as dp

from .constants import AI_ASSISTANT_NAME

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter OpenAI API key: ")

if not os.environ.get("TOGETHER_API_KEY"):
    os.environ["TOGETHER_API_KEY"] = getpass.getpass(prompt="Enter Together API key: ")

ai_assistant = dp.DspyAiAssistant(
    name=AI_ASSISTANT_NAME,
    models=dp.Models(
        task="together_ai/google/gemma-3n-e4b-it",
        teacher="openai/gpt-5",
        embedding="openai/text-embedding-3-small",
    ),
    corpus=dp.Document(
        name="European Central Bank Supervisory Manual",
        pages=dp.utils.load_pdf(path="documents/ecb_supervisory_manual.pdf"),
    ),
    dataset=dp.Examples(
        qa_pairs=dp.utils.load_json(path="examples/ecb_supervisory_manual.json"),
    ),
    metrics={
        "recall": dp.metrics.SimpleRecall(judge_model="openai/gpt-5"),
        "sources": dp.metrics.SimpleSourcesCoverage(judge_model="openai/gpt-5"),
    },
)

if __name__ == "__main__":
    ai_assistant.ask(question="What does SSM stand for?")
    ai_assistant.evaluate(metric="recall")
    ai_assistant.optimize(metric="recall", effort="light")
