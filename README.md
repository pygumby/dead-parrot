<div align="center">

# 😵🦜 dead-parrot
## A Framework for Scalable AI Agents

[🐱 GitHub](https://github.com/pygumby/dead-parrot) |
[🐍 PyPI](https://pypi.org/project/dead-parrot)

</div>

dead-parrot is a Python package created by Lucas Konstantin Bärenfänger ([@pygumby](https://github.com/pygumby)) as part of his thesis for the master's program "Data Analytics & Management" at Frankfurt School of Finance & Management.
The thesis addresses the challenges of maintaining AI agents at scale, as experienced at the European Central Bank (ECB).
These challenges include heterogeneous technology stacks, sensitivity to the choice of underlying language models (LMs) and more.
dead-parrot implements the approaches identified to address these challenges.

----

### Usage

dead-parrot is available on [PyPI](https://pypi.org/project/dead-parrot) and can be installed via `uv add dead-parrot` or `pip install dead-parrot`.

The [demos](https://github.com/pygumby/dead-parrot/tree/main/demos/) folder contains three demos:

- [ecb_bs_expert_agent](https://github.com/pygumby/dead-parrot/tree/main/demos/ecb_bs_expert_agent/): An expert agent that answers questions on ECB Banking Supervision matters.
- [ecb_hr_expert_agent](https://github.com/pygumby/dead-parrot/tree/main/demos/ecb_bs_expert_agent/): An expert agent that answers questions on ECB Human Resources matters.
- [ecb_triage_agent](https://github.com/pygumby/dead-parrot/tree/main/demos/ecb_bs_expert_agent/): A triage agent that answers questions on ECB matters by interacting with expert agents.

While the two expert agents can be run standalone, together, they conform to the general architecture of dead-parrot:

```
                         ┌──────────┐┌─────────┐
                         │   MCP    ││  REST   │
                         │  server  ││ server  │
                         └──────────┘└─────────┘
                         ┌─────────────────────┐
                         │ ┌─────────────────┐ │
                         │ │  Triage agent   │ │
                         │ │     (ReAct)     │ │
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
│ │      (RAG)      │ │  │ │      (RAG)      │ │  │ │      (RAG)      │ │
│ └─────────────────┘ │  │ └─────────────────┘ │  │ └─────────────────┘ │
│  Temporal workflow  │  │  Temporal workflow  │  │  Temporal workflow  │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

All three demos are fully self-contained projects that can serve as templates.
To start them all together, ensure uv and Temporal are installed, then run `./demos/start_all.sh`.
To interact with the triage agent, hook up `localhost:9000/mcp` with your favorite local MCP client, e.g., [Goose](https://goose-docs.ai) or [Claude Desktop](https://claude.ai/downloads).
Please refer to each demo's README.md for more information.

### Development

- Install dependencies: `uv sync --all-packages`
- Type-check: `uv run mypy .`
- Lint: `uv run ruff check . --fix`
- Format: `uv run ruff format .`

### License

[MIT License](https://github.com/pygumby/dead-parrot/blob/main/LICENSE)
