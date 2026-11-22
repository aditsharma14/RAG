# Self-Reflective RAG with LangGraph: Groq for Generation and Critique, Hugging Face for Retrieval

This folder builds a RAG pipeline that **checks its own answers**. After the LLM answers, a second LLM call acts as a **critic**. It compares the answer with the retrieved context and decides whether every claim is supported and whether the question is fully answered. If not, the critic's feedback is used to rewrite the search query, retrieve again and produce a better answer. The loop stops when the critic accepts the answer or after 3 attempts.

```
SelfReflection/
├── code.ipynb           # Notebook: FAISS index, retrieve → answer → reflect → rewrite loop
├── internal_docs.txt    # Small fictional "production stack" note (created by the notebook if missing)
└── readme.md            # This file
```

---

## 1. Core Concepts

### Why self-reflection?
A plain RAG pipeline trusts its first answer. The LLM can still:

- **hallucinate**: add facts that are not in the retrieved context,
- **miss parts** of the question,
- **answer from bad context** when retrieval found the wrong chunks.

Self-reflection adds a review step that catches these problems before the answer is returned. The research notes in the [React-Rag](../React-Rag/readme.md) module describe this idea (Experiment 16): a reflection step cut hallucinations by 30% at the cost of extra latency.

### The graph

```
START ──► retriever ──► responder ──► reflector ──is_good or attempts ≥ 3?──yes──► done ──► END
              ▲                                          │
              │                                          no
              │                                          ▼
              └────────────────────────────────────── rewriter
```

| Node | Job |
|---|---|
| `retriever` | Searches FAISS with `search_query` (or the original question on the first pass) |
| `responder` | Answers **only from the context**. On retries, it is also shown the critic's previous feedback |
| `reflector` | The critic: returns `grounded`, `complete` and `feedback` as structured output |
| `rewriter` | Turns the critic's feedback into a new search query, so the retry actually retrieves something different |
| `done` | Empty node that marks the exit point before `END` |

### What the critic checks
```python
class Reflection(BaseModel):
    grounded: bool   # every claim in the answer is supported by the context
    complete: bool   # the answer fully addresses the question
    feedback: str    # what is missing or unsupported, and what to search for next
```

The answer is accepted only if **both** `grounded` and `complete` are true. Two design choices make the critic useful:

1. **It sees the context.** The critic judges the answer against the retrieved documents, not against its own knowledge. Without the context, it could only judge whether the answer *sounds* right.
2. **Its feedback drives the retry.** The feedback is used twice: by `rewriter` to search for the missing information, and by `responder` to fix the specific problem. A retry with the same query and no feedback would most likely produce the same answer.

### Stopping the loop
`MAX_ATTEMPTS = 3` caps the number of answer attempts. Without a cap, a question the documents cannot answer would loop forever, because the critic would never mark the answer `complete`.

---

## 2. Setup

### API keys
`.env` in the repository root (one level above this folder):

```env
groq_api=your_groq_api_key
hf_token=your_hugging_face_token
```

Cell 2 finds the file with `find_dotenv(usecwd=True)` and copies the lowercase keys into the names the libraries read (`GROQ_API_KEY`, `HF_TOKEN`). If `GROQ_API_KEY` is never set, `ChatGroq` fails with *"The api_key client option must be set..."*.

### Packages
All are in the repository's `requirements.txt`: `langchain-groq`, `langchain-huggingface`, `sentence-transformers`, `langchain-community`, `faiss-cpu`, `langgraph`, `python-dotenv`. `pydantic` and `langchain-text-splitters` come in as dependencies.

---

## 3. Notebook Walkthrough

### Cell 1: Imports
`Document`, `RecursiveCharacterTextSplitter`, `TextLoader`, `FAISS`, `ChatGroq`, `HuggingFaceEmbeddings`, `StateGraph`, `END`, plus `BaseModel` / `Field` for the state and the critic's output.

### Cell 2: API keys
Loads `.env` from the repository root, maps the keys and asserts that `GROQ_API_KEY` is set. **Output:** `GROQ_API_KEY loaded: True`.

### Cell 3: LLM and embeddings

```python
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

The same LLM plays three roles: responder, critic and query rewriter. `temperature=0` keeps the critic's verdicts repeatable. The original example used `llama-3.3-70b-versatile`, which Groq no longer serves (`404 model_not_found`).

### Cell 4: The knowledge base
Writes `internal_docs.txt` only if it does not exist yet. The note lists the models in a fictional production stack: **EfficientFormer**, **Reformer**, **LLaMA2** and **TinyBERT** are in use. **DistilBERT** was evaluated but replaced. It also mentions the Triton inference server.

The DistilBERT line is a useful trap: a careless answer would list it as a production model.

### Cell 5: Vector store
Loads the file, splits it into 500-character chunks with 50 characters of overlap, and indexes them in FAISS with `k=4`. **Output:** `Chunks indexed: 2`. The file is only ~540 characters long, so every query retrieves the whole document.

### Cell 6: State and the critic

| Field | Written by | Meaning |
|---|---|---|
| `question` | Caller | The user's question, never changed |
| `search_query` | `rewriter` | Query used for retrieval on retries |
| `retrieved_docs` | `retriever` | Current chunks |
| `answer` | `responder` | Latest answer |
| `reflection` | `reflector` | Critic feedback |
| `is_good` | `reflector` | `grounded and complete` |
| `attempts` | `responder` | Number of answers generated so far |

`critic = llm.with_structured_output(Reflection)` makes the critic return a `Reflection` object, so its verdict is read as booleans without any string parsing.

### Cell 7: Nodes
- **`retrieve_docs`**: uses `search_query or question` and prints the query it used.
- **`generate_answer`**: "Answer using ONLY the context... If the context does not contain the answer, say so clearly." On retries (`attempts > 0`) it adds the critic's feedback with "Fix it." Increments `attempts`.
- **`reflect_on_answer`**: sends the context, question and answer to the critic and prints `grounded` / `complete` for the attempt.
- **`rewrite_query`**: asks for a new search query based on the question and the feedback, and strips surrounding quotes from the reply.
- **`finalize`**: returns `{}`; the state is already complete.

### Cell 8: Graph

```python
def route_after_reflection(state):
    if state.is_good or state.attempts >= MAX_ATTEMPTS:
        return "done"
    return "rewriter"

builder.add_conditional_edges("reflector", route_after_reflection,
                              {"done": "done", "rewriter": "rewriter"})
builder.add_edge("rewriter", "retriever")
```

The `rewriter → retriever` edge closes the loop.

### Cell 9: Draw the graph
Renders the graph with `draw_mermaid_png()`. This calls the mermaid.ink web service, so it is wrapped in `try/except` and prints a message instead of crashing when offline.

### Cell 10: Run

```python
user_query = "What are the transformer variants in production deployments?"
result = graph.invoke(RAGReflectionState(question=user_query))
```

**Output of the saved run:**

```
🔎 Retrieving with: 'What are the transformer variants in production deployments?' -> 2 docs
🪞 Attempt 1: grounded=True, complete=True

🧠 Final Answer: EfficientFormer, Reformer, LLaMA2, TinyBERT
🔁 Last Reflection: All transformer variants in production deployments are correctly listed.
✅ Accepted by critic: True
🔄 Total Attempts: 1
```

The answer is correct and leaves out DistilBERT. The critic accepted it on the first attempt, so **the rewrite loop did not run** in this example. With a 2-chunk knowledge base, retrieval always returns everything, so the first answer has all the context it needs.

To see the loop in action, ask something the document only partly covers, for example *"Which transformer variants are in production and what hardware do they run on?"*. The critic should mark the answer `complete=False`, and you will see `rewriter` and a second `🔎 Retrieving with:` line.

---

## 4. End-to-End Data Flow

```
 internal_docs.txt ──► TextLoader ──► splitter (500/50) ──► 2 chunks ──► MiniLM ──► FAISS (k=4)

 RAGReflectionState(question)
        │
        ▼
   retriever ── search_query or question ──► retrieved_docs
        │
        ▼
   responder ── context (+ previous feedback) ──► answer, attempts += 1
        │
        ▼
   reflector ── context + question + answer ──► Reflection(grounded, complete, feedback)
        │
        ├── is_good, or attempts ≥ 3 ──► done ──► END ──► result["answer"]
        │
        └── otherwise ──► rewriter ── feedback ──► new search_query ──► retriever
```

---

## 5. Limitations and Next Steps

| Limitation | Effect | Possible improvement |
|---|---|---|
| Same model answers and critiques | The critic may share the responder's blind spots | Use a different or larger Groq model for the critic |
| Tiny knowledge base | Retrieval always returns everything, so the loop is not exercised | Add more documents, or ask a question the notes only partly answer |
| After 3 failed attempts, the last answer is returned anyway | The user may get an answer the critic rejected | Check `result["is_good"]` and show a warning, or return "I don't know" |
| `retrieved_docs` is replaced on each retry | Useful chunks from the first search are dropped | Merge new documents with the previous ones |
| Each retry costs 3 more LLM calls | Higher latency and cost | Lower `MAX_ATTEMPTS`, or skip reflection for simple questions |

---

## 6. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `GroqError: The api_key client option must be set...` | `.env` has `groq_api` but `ChatGroq` reads `GROQ_API_KEY`. Run Cell 2, which maps the name. |
| `404 model_not_found` for `llama-3.3-70b-versatile` | The model was retired on Groq. Use `openai/gpt-oss-20b`. |
| `.env found at: NOT FOUND` | The notebook is not running inside the repository. Open it from `SelfReflection/`. |
| Graph image does not render | `draw_mermaid_png()` needs internet access. Use `print(graph.get_graph().draw_mermaid())` instead. |
| Loop always runs 3 attempts | The critic keeps rejecting the answer, usually because the documents do not contain it. Print `result["reflection"]` to see why. |
