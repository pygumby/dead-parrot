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
    corpus=dp.Corpus(
        name="European Central Bank Staff Rules",
        texts=dp.utils.load_pages_from_pdf(path="context/ecb_staff_rules.pdf"),
        chunk_size=500,
        retriever_k=3,
    ),
    dataset=dp.Dataset(
        examples=dp.utils.load_dicts_from_json(path="dataset/examples.json"),
        question_key="q",
        answer_key="a",
    ),
    metrics={
        "recall": dp.metrics.SimpleRecall(judge_model="openai/gpt-5"),
        "sources": dp.metrics.SimpleSourcesCoverage(judge_model="openai/gpt-5"),
    },
)

ecb_ai_assistant.ask(question="How long is the probationary period?")
ecb_ai_assistant.evaluate(metric="recall")
ecb_ai_assistant.optimize(metric="recall", effort="light")
