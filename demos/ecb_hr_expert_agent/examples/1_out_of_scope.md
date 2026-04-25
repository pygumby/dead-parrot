# Task: Generate an evaluation dataset for a RAG system on ECB HR – Category "Out-of-scope refusals" (15 question-answer pairs)

## Context
I am building an "expert agent", i.e., a retrieval-augmented generation (RAG) system, that answers questions on European Central Bank (ECB) Human Resources (HR) matters, grounded in public ECB documents.

I attached the following such documents:
- n/a

I need a "golden" evaluation dataset of question-answer pairs to use for evaluating and optimizing the system.

## Task
Generate exactly 15 question-answer pairs based on the attached document that fall into the category "Out-of-scope refusals":
- **Expected questions**:
  Inquiries about non-HR matters, e.g., monetary policy, general knowledge, current events, IT support, personal advice unrelated to HR.
- **Expected answer**:
  A brief, formal refusal explaining that the agent only handles HR matters.
  Nothing more – do not invent referrals to other people, teams, systems or websites.
  No source attributions.

## Constraints

- **Tone**:
  Formal, professional, neutral
- **Length**:
  1 to 4 sentences maximum

## Quality

- Questions should sound like realistic queries from ECB staff members, not like exam questions.
- Do not duplicate questions or create near-duplicates.
- Vary the topics so the refusal behavior is tested broadly, not just on one type of off-topic question.

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
