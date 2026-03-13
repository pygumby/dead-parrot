"""Demo script showcasing how to create, query and evaluate an AI Assistant."""

import getpass
import os

import dotenv

import dead_parrot as dp

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter API key for OpenAI: ")

if not os.environ.get("TOGETHER_API_KEY"):
    os.environ["TOGETHER_API_KEY"] = getpass.getpass(
        prompt="Enter API key for Together: "
    )

ecb_ai_assistant: dp.AiAssistant = dp.DspyAiAssistant(
    name="ECB AI Assistant",
    task_model="together_ai/google/gemma-3n-e4b-it",
    teacher_model="openai/gpt-4o",
    embedding_model="openai/text-embedding-3-small",
    corpus=dp.utils.load_corpus_from_pdf(
        name="European Central Bank Staff Rules",
        path="context/ecb_staff_rules.pdf",
    ),
    examples=dp.utils.load_examples_from_json(
        path="examples/ecb_staff_rules.json",
    ),
)

ecb_ai_assistant.ask(question="How long is the probationary period?")
ecb_ai_assistant.evaluate()
ecb_ai_assistant.optimize()
