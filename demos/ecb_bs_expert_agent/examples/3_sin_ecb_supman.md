# Task: Generate an evaluation dataset for a RAG system on ECB BS – Category "Single-passage ECB Supervisory Manual" (30 question-answer pairs)

## Context
I am building an "expert agent", i.e., a retrieval-augmented generation (RAG) system, that answers questions on European Central Bank (ECB) Banking Supervision (BS) matters, grounded in public ECB documents.

I attached the following such documents:
- Document name: "ECB Supervisory Manual", file name: `ecb_supman.pdf`

I need a "golden" evaluation dataset of question-answer pairs to use for evaluating and optimizing the system.

## Task
Generate exactly 30 question-answer pairs that fall into the category "Single-passage ECB Supervisory Manual":
- **Expected questions**:
  BS-related inquiries from ECB staff members whose correct answer is fully contained in one passage of the attached document, e.g., one paragraph or a contiguous section.
- **Expected answers**:
  Answers grounded strictly in one passage, with a single citation.
  Answers must not rely on outside knowledge or inference beyond what the passage supports.

## Constraints

- **Tone**:
  Formal, professional, neutral
- **Length**:
  1 to 4 sentences maximum
- **Grounding**:
  Every factual claim must be directly supported by the cited passage.
  Do not paraphrase beyond what the text says.
  Do not add interpretation.
- **Single-passage requirement**:
  For each question, the answer must be fully contained in one passage.
  If answering well requires combining information from multiple paragraphs or sections, the question does not belong in this category.
- **Citation format**:
  Append the citation at the end of the answer in parentheses, in this exact form:
  - `(Document name, page N)`

## Quality

- Questions should sound like realistic queries from ECB staff members, not like exam questions.
- Do not duplicate questions or create near-duplicates.
- Cover a diverse range of BS topics present in the document.
  Before generating questions, briefly survey the document and identify the major BS topic areas it covers.
  Distribute questions across these areas rather than clustering on whichever topic appears first or most prominently.
- Distribute citations across the document.
  Do not concentrate questions in the first few pages, ensure the later sections of the document are also represented proportionally to their content.
- Favor questions where the answer hinges on a specific, verifiable detail, e.g., a number, a deadline, a condition, a defined term.
  These are higher-value for evaluation than questions with vague or paraphrased answers.

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
