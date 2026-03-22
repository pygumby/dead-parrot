<div align="center">

# 😵🦜 dead-parrot
## A Framework for Scalable AI Assistants

[🐱 GitHub](https://github.com/pygumby/dead-parrot/) |
[🐍 PyPI](https://pypi.org/project/dead-parrot/)

</div>

dead-parrot is a Python package created by Lucas Konstantin Bärenfänger ([@pygumby](https://github.com/pygumby)) as part of his thesis for the master's program "Data Analytics & Management" at Frankfurt School of Finance & Management.
The thesis addresses the challenges of maintaining AI assistants at scale, as experienced at the European Central Bank (ECB).
These challenges include heterogeneous technology stacks, sensitivity to the choice of underlying language models (LMs) and more.
dead-parrot implements the approaches identified to address these challenges.

----

### Usage

dead-parrot is available on [PyPI](https://pypi.org/project/dead-parrot/) and can be installed via `uv add dead-parrot` or `pip install dead-parrot`.

dead-parrot enables the creation of a simple AI assistant in as little as ~10 lines of code:

```python
import dead_parrot as dp

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
    metric=dp.metrics.SimpleRecall(
        judge_model="openai/gpt-4o",
    ),
)

ecb_ai_assistant.ask(question="How long is the probationary period?")
ecb_ai_assistant.evaluate()
ecb_ai_assistant.optimize()
ecb_ai_assistant.evaluate()
```

Please refer to the [demos](https://github.com/pygumby/dead-parrot/tree/main/demos/) folder in the repository for fully self-contained usage examples.

### Development

- Install dependencies: `uv sync`
- Type-check: `uv run mypy .`
- Lint: `uv run ruff check . --fix`
- Format: `uv run ruff format .`

### License

[MIT License](https://github.com/pygumby/dead-parrot/blob/main/LICENSE)
