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

dead-parrot enables the creation of a simple AI assistant a few lines of code:

```python
import dead_parrot as dp

ecb_ai_assistant: dp.AiAssistant = dp.DspyAiAssistant(
    name="ECB AI Assistant",
    models=dp.Models(
        task="together_ai/google/gemma-3n-e4b-it",
        teacher="openai/gpt-5",
        embedding="openai/text-embedding-3-small",
    ),
    corpus=[
        dp.Document(
            name="European Central Bank Staff Rules",
            texts=dp.utils.load_texts_from_pdf(path="corpus/ecb_rules.pdf"),
            chunk_size=500,
        ),
        dp.Document(
            name="European Central Bank Conditions of Employment",
            texts=dp.utils.load_texts_from_pdf(path="corpus/ecb_conditions.pdf"),
            chunk_size=500,
        ),
    ],
    dataset=dp.Dataset(
        examples=dp.utils.load_dicts_from_json(path="dataset/examples.json"),
        question_key="q",
        answer_key="a",
    ),
    metrics={
        "recall": dp.metrics.SimpleRecall(judge_model="openai/gpt-5"),
        "sources": dp.metrics.SimpleSourcesCoverage(judge_model="openai/gpt-5"),
    },
)

ecb_ai_assistant.ask(question="How long is the probationary period?")
ecb_ai_assistant.evaluate(metric="recall")
ecb_ai_assistant.optimize(metric="recall", effort="light")
```

Please refer to the [demos](https://github.com/pygumby/dead-parrot/tree/main/demos/) folder in the repository for fully self-contained usage examples.

### Development

- Install dependencies: `uv sync`
- Type-check: `uv run mypy .`
- Lint: `uv run ruff check . --fix`
- Format: `uv run ruff format .`

### License

[MIT License](https://github.com/pygumby/dead-parrot/blob/main/LICENSE)
