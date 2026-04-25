# Task: Generate an evaluation dataset for a RAG system on ECB HR – Category "Multi-passage ECB Conditions of Employment" (5 question-answer pairs)

## Context
I am building an "expert agent", i.e., a retrieval-augmented generation (RAG) system, that answers questions on European Central Bank (ECB) Human Resources (HR) matters, grounded in public ECB documents.

I attached the following such documents:
- Document name: "ECB Conditions of Employment", file name: `ecb_conds.pdf`

I need a "golden" evaluation dataset of question-answer pairs to use for evaluating and optimizing the system.

## Task
Generate exactly 5 question-answer pairs that fall into the category "Multi-passage ECB Conditions of Employment":
- **Expected questions**:
  HR-related inquiries from ECB staff members whose correct answer requires combining information from two or more distinct passages within the attached document, e.g., paragraphs on different pages, or different articles/sections.
  No single passage is sufficient on its own.
- **Expected answer**:
  Answer grounded strictly in the cited passages, with citations pointing to every passage used.
  The answer must not rely on outside knowledge or inference beyond what the passages support.

## Constraints

- **Tone**:
  Formal, professional, neutral
- **Length**:
  1 to 4 sentences maximum
- **Grounding**:
  Every factual claim must be directly supported by one of the cited passages.
  Do not paraphrase beyond what the text says.
  Do not add interpretation.
- **Multi-passage requirement**:
  For each question, before writing the answer, identify the specific passages required.
  If the question can be fully answered from one passage alone, it does not belong in this category.
  The passages must be genuinely distinct — different paragraphs, sections, or articles — not contiguous sentences from the same paragraph.
- **Citation format**:
  Append the citation at the end of the answer in parentheses, in this exact form:
  - Multiple distinct pages: `(Document name, pages N, M)`
  - A range: `(Document name, pages N–M)`

## Quality

- Questions should sound like realistic queries from ECB staff members, not like exam questions
  Favor scenario-based phrasing where the cross-passage dependency arises naturally from the user's situation.
- Do not duplicate questions or create near-duplicates.
  Do not generate multiple questions that combine the same pair of passages.
- Aim for variety in *how* passages combine.
- Distribute citations across the document. Do not concentrate questions in the first few pages or any single section; the later parts of the document should also be represented.

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
