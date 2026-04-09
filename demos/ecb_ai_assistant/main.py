"""Demo script showcasing how to create, query and evaluate an AI Assistant."""

import getpass
import os

import dotenv

import dead_parrot as dp

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter OpenAI API key: ")

if not os.environ.get("TOGETHER_API_KEY"):
    os.environ["TOGETHER_API_KEY"] = getpass.getpass(prompt="Enter Together API key: ")

ecb_ai_assistant: dp.AiAssistant = dp.DspyAiAssistant(
    name="ECB AI Assistant",
    models=dp.Models(
        task="together_ai/google/gemma-3n-e4b-it",
        teacher="openai/gpt-5",
        embedding="openai/text-embedding-3-small",
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
        "recall": dp.metrics.SimpleRecall(judge_model="openai/gpt-5"),
        "sources": dp.metrics.SimpleSourcesCoverage(judge_model="openai/gpt-5"),
    },
)

ecb_ai_assistant.ask(question="How long is the probationary period?")
ecb_ai_assistant.evaluate(metric="recall")
ecb_ai_assistant.optimize(metric="recall", effort="light")
