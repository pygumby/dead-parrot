<div align="center">

# 😵🦜 dead-parrot
## A Framework for Scalable AI Agents

[🐱 GitHub](https://github.com/pygumby/dead-parrot) |
[🐍 PyPI](https://pypi.org/project/dead-parrot)

</div>

dead-parrot is a Python package created by Lucas Konstantin Bärenfänger ([@pygumby](https://github.com/pygumby)) as part of his thesis for the master's program "Data Analytics & Management" at Frankfurt School of Finance & Management.
The thesis addresses the challenges of building and maintaining AI agents at scale in large organizations.
These include heterogeneous technology stacks, sensitivity to the choice of underlying language models (LMs) and more.
dead-parrot implements the approaches proposed to meet the challenges.

----

### Overview

dead-parrot defines two key abstractions:
- **Expert agents** answer questions based on subject-matter expertise, via retrieval-augmented generation (RAG)
- **Triage agents** answer questions by interacting with expert agents in an agentic loop, via the ReAct pattern

dead-parrot builds on three key technologies:
- [DSPy](https://dspy.ai) to program LM pipelines declaratively, replacing brittle prompt engineering
- [Temporal](https://temporal.io) to orchestrate agents as durable workflows, with automatic retries and recovery
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io) to expose agents in a standardized way, enabling integration with LM clients

```
                         ┌──────────┐┌─────────┐
                         │   MCP    ││  REST   │
                         │  server  ││ server  │
                         └──────────┘└─────────┘
                         ┌─────────────────────┐
                         │ ┌─────────────────┐ │
                         │ │  Triage agent   │ │
                         │ │  (DSPy ReAct)   │ │
                         │ └─────────────────┘ │
                         │  Temporal workflow  │
                         └─────────────────────┘
                         ┌─────────────────────┐
                         │    REST clients     │
                         └─────────────────────┘
                                    ▲
                                    │
                                    │
                 ┌──────────────────┴─────┬────────────────────────┐
                 │                        │                        │
                 │                        │                        │
                 ▼                        ▼                        ▼
┌──────────┐┌─────────┐  ┌──────────┐┌─────────┐  ┌──────────┐┌─────────┐
│   MCP    ││  REST   │  │   MCP    ││  REST   │  │   MCP    ││  REST   │
│  server  ││ server  │  │  server  ││ server  │  │  server  ││ server  │
└──────────┘└─────────┘  └──────────┘└─────────┘  └──────────┘└─────────┘
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ ┌─────────────────┐ │  │ ┌─────────────────┐ │  │ ┌─────────────────┐ │
│ │ Expert agent #1 │ │  │ │ Expert agent #2 │ │  │ │ Expert agent #n │ │
│ │   (DSPy RAG)    │ │  │ │   (DSPy RAG)    │ │  │ │   (DSPy RAG)    │ │
│ └─────────────────┘ │  │ └─────────────────┘ │  │ └─────────────────┘ │
│  Temporal workflow  │  │  Temporal workflow  │  │  Temporal workflow  │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

### Usage

dead-parrot is available on [PyPI](https://pypi.org/project/dead-parrot) and can be installed via `uv add dead-parrot` or `pip install dead-parrot`.

The [demos](https://github.com/pygumby/dead-parrot/tree/main/demos/) folder contains three projects that serve as examples of and templates for building agents with dead-parrot.
There are two expert agents grounded in public documents from the European Central Bank (ECB), and one triage agent that answers questions by interacting with them:

- [ecb_bs_expert_agent](https://github.com/pygumby/dead-parrot/tree/main/demos/ecb_bs_expert_agent/): Expert agent for ECB Banking Supervision matters, grounded in public ECB documents
- [ecb_hr_expert_agent](https://github.com/pygumby/dead-parrot/tree/main/demos/ecb_hr_expert_agent/): Expert agent for ECB Human Resources matters, grounded in public ECB documents
- [ecb_triage_agent](https://github.com/pygumby/dead-parrot/tree/main/demos/ecb_triage_agent/): Triage agent that answers questions on ECB matters by interacting with the expert agents

Each expert agent can be run standalone, but together with the triage agent they form the full architecture shown above.
To start them all together, ensure uv and Temporal are installed, then run `./demos/start_all.sh`.
To ask questions to the triage agent, hook up `localhost:9000/mcp` with your favorite local LM client, e.g., [Goose](https://goose-docs.ai) or [Claude Desktop](https://claude.ai/downloads).
Please refer to each demo's README.md for more information.

### Development

- Install dependencies: `uv sync --all-packages`
- Type-check: `uv run mypy .`
- Lint: `uv run ruff check . --fix`
- Format: `uv run ruff format .`

### License

[MIT License](https://github.com/pygumby/dead-parrot/blob/main/LICENSE)
