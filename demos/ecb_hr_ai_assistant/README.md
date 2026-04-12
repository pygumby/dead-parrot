# ecb-hr-ai-assistant

This is a demo that showcases how to build an AI assistant with [dead-parrot](https://github.com/pygumby/dead-parrot).
It implements an AI assistant that answers questions on ECB HR matters.
At the same time, it is a template that serves as a starting point for building your own AI assistant.

The structure is simple:
- [documents/](documents/) contains the corpus for context retrieval.
- [examples/](examples/) contains the dataset for evaluation and optimization.
- [constants.py](src/ecb_hr_ai_assistant/constants.py) defines names and other constants.
- [ai_assistant.py](src/ecb_hr_ai_assistant/ai_assistant.py) defines the AI assistant using dead-parrot.
- [temporal_workflow.py](src/ecb_hr_ai_assistant/temporal_workflow.py) defines a Temporal workflow that calls the AI assistant.
- [temporal_worker.py](src/ecb_hr_ai_assistant/temporal_worker.py) defines and runs a Temporal worker for the Temporal workflow.
- [rest_api.py](src/ecb_hr_ai_assistant/rest_api.py) defines and runs a FastAPI REST API that calls the Temporal workflow.
----

### Usage

Prerequisites:
- Create an .env file based on the [.env.example](.env.example) file, containing:
  - API keys, e.g., [OpenAI](https://platform.openai.com)
  - Service URLs, e.g., [Temporal](https://temporal.io)

Run the following commands from the repository's root:
1. Install dependencies:
    ```
    uv sync --all-packages
    ```
2. Optionally, query, evaluate and optimize the AI assistant:
    ```
    uv run --directory demos/ecb_hr_ai_assistant python -m ecb_hr_ai_assistant.ai_assistant
    ```
3. Start the Temporal Service (if not yet running):
    ```
    temporal server start-dev
    ```
4. Start the Temporal worker:
    ```
    uv run --directory demos/ecb_hr_ai_assistant python -m ecb_hr_ai_assistant.temporal_worker
    ```
5. Start the REST API:
    ```
    uv run --directory demos/ecb_hr_ai_assistant uvicorn ecb_hr_ai_assistant.rest_api:app --port 8001
    ```
6. Call the `/card` endpoint to get the AI assistant's card:
    ```
    curl --request GET \
    --url http://localhost:8001/card \
    --header 'content-type: application/json'
    ```
7. Call the `/ask` endpoint to ask a question to the AI assistant:
    ```
    curl --request POST \
    --url http://localhost:8001/ask \
    --header 'content-type: application/json' \
    --data '{
        "question": "What is a fixed-term contract?"
    }'
    ```

### Development

To create your own AI assistant, change the following files:
- [pyproject.toml](pyproject.toml): Update the package name.
- [constants.py](src/ecb_hr_ai_assistant/constants.py): Update the constants.
- [ai_assistant.py](src/ecb_hr_ai_assistant/ai_assistant.py): Customize the AI assistant's definition.
- [temporal_workflow.py](src/ecb_hr_ai_assistant/temporal_workflow.py): Change the Temporal worklow's name (use VS Code's "Rename Symbol" to avoid stale references).
