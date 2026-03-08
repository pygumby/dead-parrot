"""Demo script showcasing how to create, query and evaluate an AI Assistant."""

import getpass
import os

import dotenv

import dead_parrot as dp

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

ecb_ai_assistant: dp.AiAssistant = dp.DspyAiAssistant(
    lm="openai/gpt-4o-mini",
    embedder="openai/text-embedding-3-small",
    corpus=dp.utils.load_corpus_from_pdf(
        "European Central Bank Staff Rules",
        "context/ecb_staff_rules.pdf",
    ),
    examples=dp.utils.load_examples_from_json("examples/ecb_staff_rules.json"),
)
print(ecb_ai_assistant.ask("How long is the probationary period?"))
print(ecb_ai_assistant.eval())
