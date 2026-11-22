# Iterative Retrieval with Self-Reflection: Groq for Generation and Critique, Hugging Face for Retrieval

This folder builds a RAG loop that **retrieves more evidence when its answer is not good enough**. After each answer, a critic checks it against the retrieved context. If information is missing, the critic's notes are turned into a short new search query, more documents are retrieved and **added to** the ones already found, and the answer is written again. The loop stops when the critic verifies the answer or after 3 attempts.

```
IterativeRetrieval/
├── code.ipynb           # Notebook: FAISS index, retrieve → answer → reflect → refine loop
├── internal_docs.txt    # Fictional engineering report on transformer models in production (written by the notebook)
└── readme.md            # This file
```

---

## 1. Core Concepts

### Iterative retrieval
One search often misses part of what a question needs. Iterative retrieval treats retrieval as a loop:

```
retrieve ──► answer ──► reflect: what is missing? ──► new focused query ──► retrieve more ──► ...
```

Each pass **adds** evidence instead of replacing it, so the answer is built on everything found so far.

### The graph

```
START ──► retrieve ──► answer ──► reflect ──verified or attempts ≥ 3?──yes──► END
             ▲                                    │
             │                                    no
             │                                    ▼
             └─────────────────────────────── refine
```

| Node | Job |
|---|---|
| `retrieve` | Searches with `refined_question` (or the original question first), **appends** only new chunks, logs the query |
| `answer` | Answers **only from the context**, says clearly what is missing, and includes the critic's feedback on retries |
| `reflect` | Critic: returns `grounded`, `complete`, `missing_info` as structured output |
| `refine` | Writes **one short keyword query** for the missing information, avoiding queries already tried |

### How this differs from the SelfReflection module
[SelfReflection](../SelfReflection/readme.md) uses the same answer → critique → retry loop. This version changes how retries search:

| | SelfReflection | IterativeRetrieval |
|---|---|---|
| Documents on retry | **Replaced** by the new search | **Accumulated**, with de-duplication |
| New query | Free text from the LLM | Structured `SearchQuery`, max 15 words, keyword style |
| Repeated searches | Possible | Prevented: `queries_tried` is passed to the query writer |
| "Not in the context" answers | Can be marked incomplete and retried | Explicitly counted as **grounded** |
| After the attempt limit | Returns the last answer | Prints what is still missing |

### Why each design choice matters

| Choice | Reason |
|---|---|
| Accumulate documents | Evidence found in attempt 1 is not lost when attempt 2 searches for something else |
| Short keyword query | A long, question-like query embeds close to the original question and retrieves the same chunks again |
| `queries_tried` | Without it, the query writer often produces the same query again and the loop makes no progress |
| Critic sees the context | It judges support in the documents, not whether the answer *sounds* right |
| Honest "not covered" counts as grounded | Otherwise a question the documents cannot answer would always use all 3 attempts |
| `MAX_ATTEMPTS = 3` | Caps cost and stops the loop from running forever |

---

## 2. Setup

### API keys
`.env` in the repository root:

```env
groq_api=your_groq_api_key
hf_token=your_hugging_face_token
```

Cell 2 copies `groq_api` into `GROQ_API_KEY`, the name `ChatGroq` reads.

### Packages
All are in the repository's `requirements.txt`: `langchain-groq`, `langchain-huggingface`, `sentence-transformers`, `langchain-community`, `faiss-cpu`, `langgraph`, `python-dotenv`. `pydantic` and `langchain-text-splitters` come in as dependencies.

---

## 3. Notebook Walkthrough

### Cell 1: Overview
A markdown summary of the five steps of the loop.

### Cell 2: API key

```python
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("groq_api")
```

If `groq_api` is missing from `.env`, `os.getenv` returns `None` and this line raises `TypeError: str expected, not NoneType`. The other modules use `os.getenv("GROQ_API_KEY") or os.getenv("groq_api", "")` to avoid that.

### Cell 3: Imports
`Document`, `RecursiveCharacterTextSplitter`, `TextLoader`, `FAISS`, `ChatGroq`, `HuggingFaceEmbeddings`, `StateGraph`, `END`, plus `BaseModel` / `Field`.

### Cell 4: LLM and embeddings

```python
GROQ_MODEL = "openai/gpt-oss-20b"
llm = ChatGroq(model=GROQ_MODEL, temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

`ChatGroq` takes the bare model name. The `groq:` prefix is only for `init_chat_model`.

### Cell 5: The knowledge base
Writes `internal_docs.txt`, a fictional engineering report with these sections:

| Section | Contents |
|---|---|
| Transformer variants | EfficientFormer (on-device vision), DistilBERT (semantic search, being phased out for intent tasks), Reformer (memory tests), LLaMA2 (chat, with LangGraph orchestration), TinyBERT (email triage) |
| Deployment architecture | Triton with dynamic batching, EKS with A10G GPUs, Fargate for CPU models, nightly FAISS rebuilds, Prometheus/Grafana, p95 < 300 ms |
| Optimizations | Distillation, adapters, AMP on A100s, DVC/MLflow, canary rollouts, GitHub Actions → Docker → EKS |
| Additional notes | ViT/Swin, Wav2Vec 2.0, tokenizer comparisons, FlashAttention and RoPE |

The file is **overwritten on every run**, so edit the string in this cell, not the `.txt` file. **Output:** `internal_docs.txt created: 2775 characters`.

### Cell 6: Vector store
500-character chunks with 50 characters of overlap, FAISS, `k=4`. **Output:** `Chunks indexed: 8`, so one search returns half the document.

### Cell 7: State and schemas

| `IterativeRAGState` field | Meaning |
|---|---|
| `question` | Original question, never changed |
| `refined_question` | Latest query from `refine` |
| `queries_tried` | Every query used so far |
| `retrieved_docs` | **All** unique chunks found across attempts |
| `answer` | Latest answer |
| `feedback` | Critic's `missing_info` |
| `verified` | `grounded and complete` |
| `attempts` | Number of answers generated |

`critic = llm.with_structured_output(Verdict)` and `query_writer = llm.with_structured_output(SearchQuery)` both return typed objects, so nothing is parsed from free text.

### Cells 8–11: Nodes
- **`retrieve_docs`**: compares new results with the chunks already in state, adds only new ones, and logs `new docs (total N)`.
- **`generate_answer`**: three rules (no outside knowledge, no invented connections, say what is missing). On retries it includes the critic's feedback. Increments `attempts`.
- **`reflect_on_answer`**: tells the critic to judge only against the context, and that an honest "the context does not cover X" counts as grounded.
- **`refine_query`**: asks for one short query from the question, the missing information and the list of queries already tried.

### Cell 12: Graph

```python
def route_after_reflect(state):
    if state.verified or state.attempts >= MAX_ATTEMPTS:
        return "end"
    return "refine"
```

`answer` has exactly one outgoing edge (to `reflect`), and `refine → retrieve` closes the loop.

### Cell 13: Diagram
`draw_mermaid_png()` inside `try/except`, since it needs internet access.

### Cell 14: Run

```python
query = "agent loops and transformer-based systems?"
final = graph.invoke(IterativeRAGState(question=query))
```

Prints the answer, `verified`, the number of attempts, every query tried, and, if the answer was not verified, what is still missing.

**Output of the saved run:**

```
🔎 Query: 'agent loops and transformer-based systems?' -> 4 new docs (total 4)
🪞 Attempt 1: grounded=True, complete=True

✅ Final Answer: The provided context does not contain any information about "agent loops"
   or how they relate to transformer-based systems. Therefore, I cannot answer that question.
🧠 Verified: True
🔁 Attempts: 1
```

This run shows a **weakness of the critic**, not the loop working:

1. The question is vague: two topics joined by "and", with no clear ask.
2. The report has a lot about **transformer-based systems**, but the model read the question as "how do agent loops relate to transformers", found nothing on agent loops, and refused to answer **any** of it.
3. The critic marked this `grounded=True` (correct, as the refusal makes no unsupported claims) and also `complete=True` (wrong, because the transformer half was answerable). The rule "an honest 'not covered' answer is grounded" made the critic too lenient about completeness as well.
4. Because the answer was accepted on attempt 1, **the refine loop never ran**.

To see the loop at work, ask something specific that one search of 4 chunks cannot fully answer, for example *"Which transformer models run on CPU, how are they deployed, and how are model updates rolled out?"*. That needs the CPU-only models, the Fargate setup and the canary-rollout line from different chunks.

---

## 4. End-to-End Data Flow

```
 Cell 5 string ──► internal_docs.txt ──► TextLoader ──► splitter (500/50) ──► 8 chunks ──► MiniLM ──► FAISS (k=4)

 IterativeRAGState(question)
        │
        ▼
   retrieve ── refined_question or question ──► + new unique chunks ──► retrieved_docs (accumulated)
        │                                                                 queries_tried += [query]
        ▼
   answer ── all chunks (+ feedback) ──► answer, attempts += 1
        │
        ▼
   reflect ── Verdict(grounded, complete, missing_info)
        │
        ├── verified, or attempts ≥ 3 ──► END
        │
        └── otherwise ──► refine ── SearchQuery (≤ 15 words, not in queries_tried) ──► retrieve
```

---

## 5. Limitations and Next Steps

| Limitation | Effect | Possible improvement |
|---|---|---|
| Critic is lenient on completeness for "not covered" answers | A partial refusal can be accepted (the saved run) | Ask the critic to list the parts of the question and check each one separately |
| Vague multi-topic questions | The answerer misreads the question | Add a planning step first ([Query-Planning](../Query-Planning/readme.md)) |
| Accumulated context only grows | Long loops build long prompts | Keep only chunks the critic or a grader marks as relevant |
| Same model answers, critiques and writes queries | Shared blind spots | Use a different model for the critic |
| Knowledge-base file rewritten on every run | Edits to `internal_docs.txt` are lost | Write the file only `if not os.path.exists(...)` |
| `os.getenv("groq_api")` without a fallback | `TypeError` when the key is missing | `os.getenv("GROQ_API_KEY") or os.getenv("groq_api", "")` |

---

## 6. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `TypeError: str expected, not NoneType` in Cell 2 | `groq_api` is not in `.env`, or `.env` was not found. Check the file in the repository root. |
| `GroqError: The api_key client option must be set...` | Cell 2 was not run before Cell 4. |
| Changes to `internal_docs.txt` disappear | Cell 5 overwrites the file. Edit the `internal_docs` string instead. |
| Loop always ends after 1 attempt | The critic accepted the first answer. Ask a more specific question that one search cannot fully answer, as in Cell 14's notes. |
| Loop always runs 3 attempts | The documents do not contain the answer. Check `final["feedback"]` and `final["queries_tried"]`. |
| Graph image does not render | `draw_mermaid_png()` needs internet. Use `print(graph.get_graph().draw_mermaid())`. |
