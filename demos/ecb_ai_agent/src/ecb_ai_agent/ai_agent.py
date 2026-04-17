"""AI Agent."""

import getpass
import os

import dotenv

import dead_parrot as dp

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter OpenAI API key: ")

ai_agent = dp.AiAgent(
    name="ECB AI Agent",
    task_model="gpt-5-mini",
    ai_assistant_clients=[
        dp.AiAssistantClient(scheme="http", host="localhost", port=8001),
        dp.AiAssistantClient(scheme="http", host="localhost", port=8002),
    ],
)

if __name__ == "__main__":
    ai_agent.ask(
        question="What does SSM stand for? How long is the probationary period?"
    )
