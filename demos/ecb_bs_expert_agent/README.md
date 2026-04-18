# ecb-bs-expert-agent

This is a demo that showcases how to build an expert agent with [dead-parrot](https://github.com/pygumby/dead-parrot).
It implements an expert agent that answers questions on ECB Banking Supervision matters.
At the same time, it is a template that serves as a starting point for building your own expert agent.

The structure is simple:
- [documents/](documents/) contains the corpus for context retrieval.
- [examples/](examples/) contains the dataset for evaluation and optimization.
- [constants.py](src/ecb_bs_expert_agent/constants.py) defines names and other constants.
- [expert_agent.py](src/ecb_bs_expert_agent/expert_agent.py) defines the expert agent using dead-parrot.
- [temporal_workflow.py](src/ecb_bs_expert_agent/temporal_workflow.py) defines a Temporal workflow that calls the expert agent.
- [temporal_worker.py](src/ecb_bs_expert_agent/temporal_worker.py) defines and runs a Temporal worker for the Temporal workflow.
- [rest_server.py](src/ecb_bs_expert_agent/rest_server.py) defines and runs a REST server that exposes the Temporal workflow.
- [mcp_server.py](src/ecb_bs_expert_agent/rest_server.py) defines and runs an MCP server that exposes the Temporal workflow.

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
2. Optionally, query, evaluate and optimize the expert agent:
    ```
    uv run --directory demos/ecb_bs_expert_agent python -m ecb_bs_expert_agent.expert_agent
    ```
3. Start the Temporal Service (if not yet running):
    ```
    temporal server start-dev
    ```
4. Start the Temporal worker:
    ```
    uv run --directory demos/ecb_bs_expert_agent python -m ecb_bs_expert_agent.temporal_worker
    ```
5. Start the REST server:
    ```
    uv run --directory demos/ecb_bs_expert_agent python -m ecb_bs_expert_agent.rest_server
    ```
    Call the `/card` REST endpoint to get the expert agent's card:
    ```
    curl --request GET \
    --url http://localhost:8001/card \
    --header 'content-type: application/json'
    ```
    Call the `/ask` REST endpoint to ask a question to the expert agent:
    ```
    curl --request POST \
    --url http://localhost:8001/ask \
    --header 'content-type: application/json' \
    --data '{
        "question": "What does SSM stand for?"
    }'
    ```
8. Optionally, start the MCP server:
    ```
    uv run --directory demos/ecb_bs_expert_agent python -m ecb_bs_expert_agent.mcp_server
    ```
    Call the MCP tool via the MCP Inspector (requires npm to be installed, as URL provide `http://localhost:9001/mcp`):
    ```
    npx @modelcontextprotocol/inspector
    ```

### Development

To create your own expert agent, change the following files:
- [pyproject.toml](pyproject.toml)
  Update the package name.
- [constants.py](src/ecb_bs_expert_agent/constants.py)
  Update the constants.
- [expert_agent.py](src/ecb_bs_expert_agent/expert_agent.py)
  Customize the expert agent's definition.
- [temporal_workflow.py](src/ecb_bs_expert_agent/temporal_workflow.py)
  Change the Temporal worklow's name and update references accordingly in:
  - [temporal_worker.py](src/ecb_bs_expert_agent/temporal_worker.py)
  - [rest_server.py](src/ecb_bs_expert_agent/rest_server.py)
  - [mcp_server.py](src/ecb_bs_expert_agent/mcp_server.py)
