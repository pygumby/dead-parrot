# Task: Generate an evaluation dataset for a RAG system on ECB HR – Category "Unanswerable" (5 question-answer pairs)

## Context
I am building an "expert agent", i.e., a retrieval-augmented generation (RAG) system, that answers questions on European Central Bank (ECB) Human Resources (HR) matters, grounded in public ECB documents.

I attached the following such documents:
- Document name: "ECB Staff Rules", file name: `ecb_rules.pdf`
- Document name: "ECB Conditions of Employment", file name: `ecb_conds.pdf`

I need a "golden" evaluation dataset of question-answer pairs to use for evaluating and optimizing the system.

## Task
Generate exactly 5 question-answer pairs that fall into the category "Unanswerable":
- **Expected questions**:
  HR-related inquiries from ECB staff members that are *not* answerable from *either* of the attached documents.
  The questions should be plausibly in scope, i.e., the kind of HR question an ECB staff member might reasonably ask the agent, but the answer is not contained in the source material.
- **Expected answer**:
  A brief, formal acknowledgement that the agent cannot answer the question based on the available source documents.
  No source attributions.

## Constraints

- **Tone**:
  Formal, professional, neutral
- **Length**:
  1 to 4 sentences maximum
- **Grounding**:
  Before writing each question, verify that neither attached document contains the answer.
  If either document covers the topic, the question does not belong in this category.

## Quality

- Questions should sound like realistic queries from ECB staff members, not like exam questions.
- Do not duplicate questions or create near-duplicates.
- Vary the *type* of unanswerability across the pairs. Aim for a mix of:
  - HR topics plausibly in scope but genuinely absent from both documents.
  - HR topics adjacent to what the documents cover but not directly addressed.
  - HR topics that sound like they should be in the source documents but aren't.
- Vary the refusal phrasing across answers so the evaluation isn't anchored to a single template.

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
