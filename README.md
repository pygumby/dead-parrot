<div align="center">

# 😵🦜 dead-parrot
## A Framework for Scalable AI Agents

[🐱 GitHub](https://github.com/pygumby/dead-parrot/) |
[🐍 PyPI](https://pypi.org/project/dead-parrot/)

</div>

dead-parrot is a Python package created by Lucas Konstantin Bärenfänger ([@pygumby](https://github.com/pygumby)) as part of his thesis for the master's program "Data Analytics & Management" at Frankfurt School of Finance & Management.
The thesis addresses the challenges of maintaining AI agents at scale, as experienced at the European Central Bank (ECB).
These challenges include heterogeneous technology stacks, sensitivity to the choice of underlying language models (LMs) and more.
dead-parrot implements the approaches identified to address these challenges.

----

### Usage

dead-parrot is available on [PyPI](https://pypi.org/project/dead-parrot/) and can be installed via `uv add dead-parrot` or `pip install dead-parrot`.

Please refer to the [demos](https://github.com/pygumby/dead-parrot/tree/main/demos/) folder for fully self-contained usage examples and templates.

### Development

- Install dependencies: `uv sync --all-packages`
- Type-check: `uv run mypy .`
- Lint: `uv run ruff check . --fix`
- Format: `uv run ruff format .`

### License

[MIT License](https://github.com/pygumby/dead-parrot/blob/main/LICENSE)
