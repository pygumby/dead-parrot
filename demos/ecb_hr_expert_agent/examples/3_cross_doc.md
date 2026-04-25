# Task: Generate an evaluation dataset for a RAG system on ECB HR – Category "Cross-document" (5 question-answer pairs)

## Context
I am building an "expert agent", i.e., a retrieval-augmented generation (RAG) system, that answers questions on European Central Bank (ECB) Human Resources (HR) matters, grounded in public ECB documents.

I attached the following such documents:
- Document name: "ECB Staff Rules", file name: `ecb_rules.pdf`
- Document name: "ECB Conditions of Employment", file name: `ecb_conds.pdf`

I need a "golden" evaluation dataset of question-answer pairs to use for evaluating and optimizing the system.

## Task
Generate exactly 5 question-answer pairs that fall into the category "Cross-document":
- **Expected questions**:
  HR-related inquiries from ECB staff members whose correct answer requires combining information from *both* attached documents.
  Neither document alone is sufficient — the answer must genuinely depend on at least one passage from each document.
- **Expected answers**:
  Answers grounded strictly in the cited passages from both documents.
  Answers must not rely on outside knowledge or inference beyond what the passages support.

## Constraints

- **Tone**:
  Formal, professional, neutral
- **Length**:
  1 to 4 sentences maximum
- **Grounding**:
  Every factual claim must be directly supported by the cited passages.
  Do not paraphrase beyond what the text says.
  Do not add interpretation.
- **Cross-document requirement**:
  For each question, before writing the answer, identify the specific passages from each document that are required.
  If the question can be fully answered from one document alone, it does not belong in this category.
- **Citation format**:
  Append citations at the end of the answer in parentheses, in this exact form:
  - `(Document #1 name, page N; Document #2 name, page M)`
  - For multiple pages within one document: `(Document #1 name, pages N, M; Document #2 name, page K)` or `(Document #1 name, pages N–M; Document #2 name, page K)` for a range

## Quality

- Questions should sound like realistic queries from ECB staff members, not like exam questions.
- Do not duplicate questions or create near-duplicates.
- Favor questions where the cross-document dependency is *natural* — i.e., a real staff member would plausibly ask this, and answering well genuinely requires both sources.

## Format

Return a single JSON array and nothing else — no prose, no markdown fences, no commentary.
The array must be valid JSON, with all strings properly escaped (use \" for internal quotes, \n for newlines if any).

Schema:
```json
[
  {
    "question": "...",
    "answer": "...",
  }
  ...
]
```
