import getpass
import os

import dotenv
import dspy

dotenv.load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

response = lm("Say 'Hello, world!'")

print(response)
