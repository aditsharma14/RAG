# Query Planning and Decomposition with LangGraph: Groq for Planning, Hugging Face for Retrieval

This folder builds a RAG pipeline for **multi-part questions**. A planner LLM splits the question into self-contained sub-questions, each sub-question is searched separately (in parallel), and the responder answers with the context **grouped by sub-question**, so every part of the question gets covered.

```
Query-Planning/
├── code.ipynb   # Notebook: web docs → FAISS, planner → parallel retriever → grouped responder
└── readme.md    # This file
```

---

## 1. Core Concepts

### The problem with one search
Take the question *"Explain how agent loops work and what are the challenges in diffusion video generation?"*. It asks about two unrelated topics. Embedded as a single query, its vector lands somewhere between the two, and the top-k results may all come from one topic, or be weak matches for both.

### Query planning
Split first, then search:

```
"How do agent loops work, and what are the challenges in diffusion video generation?"
                         │ planner
          ┌──────────────┴──────────────┐
          ▼                             ▼
 "How do agent loops work?"   "What are the challenges in diffusion video generation?"
          │ retriever (parallel)        │
          ▼                             ▼
     4 agent chunks               4 diffusion chunks
          └──────────────┬──────────────┘
                         ▼ responder
     one answer per sub-question + combined summary
```

### The graph
A linear LangGraph graph:

```
START ──► planner ──► retriever ──► responder ──► END
```

| Node | Job |
|---|---|
| `planner` | Splits the question into 2–4 self-contained sub-questions with structured output |
| `retriever` | Runs every sub-question against FAISS in parallel with `retriever.batch`, removes duplicates |
| `responder` | Builds a context section per sub-question and asks for an answer to each, plus a summary |

### What makes this version reliable

| Design choice | Why it matters |
|---|---|
| **Structured output** (`QueryPlan`) | The plan comes back as a clean Python list, with no "Here are the sub-questions:" lines or numbering to parse |
| **Self-contained sub-questions** | Each one must make sense alone, because it is embedded and searched alone. "What are its challenges?" would retrieve nothing useful |
| **"Do NOT invent connections"** rule | LLMs like to link topics ("How do agent loops relate to video diffusion?"). That sub-question matches neither topic well |
| **`retriever.batch`** | Runs all searches concurrently instead of one after another |
| **De-duplication across sub-questions** | The same chunk is never put in the prompt twice |
| **Context grouped by sub-question** | `docs_per_subq` keeps which chunks belong to which sub-question, so the responder can answer each part from its own evidence |

### Compared with the CoT module
[COT](../COT/readme.md) also splits a question before retrieving, but merges all chunks into one block. Here the grouping is kept all the way to the prompt, and the planner is told to keep independent topics separate. This fits questions that combine **independent topics**, while CoT suits a **single topic** that needs several reasoning steps.

---

## 2. Setup

### API keys
`.env` in the repository root (one level above this folder):

```env
groq_api=your_groq_api_key
hf_token=your_hugging_face_token
```

The setup cell finds the file with `find_dotenv(usecwd=True)` and copies the lowercase keys into `GROQ_API_KEY` and `HF_TOKEN`, the names `ChatGroq` and `huggingface_hub` read.

### Packages
All are in the repository's `requirements.txt`: `langchain-groq`, `langchain-huggingface`, `sentence-transformers`, `langchain-community`, `faiss-cpu`, `langgraph`, `python-dotenv`. `beautifulsoup4`, `pydantic` and `langchain-text-splitters` come in as dependencies.

---

## 3. Notebook Walkthrough

### Cell 1: Environment

```python
load_dotenv(find_dotenv(usecwd=True), override=True)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or os.getenv("groq_api", "")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") or os.getenv("hf_token", "")
assert os.environ["GROQ_API_KEY"], "groq_api / GROQ_API_KEY missing in .env"
os.environ.setdefault("USER_AGENT", "rag-learning-notebook/1.0")
```

`USER_AGENT` is set **before** `langchain_community` is imported, because `WebBaseLoader` checks it at import time. **Output:** `GROQ_API_KEY loaded: True`.

### Cell 2: Imports
`List`, `Dict`, `BaseModel`, `Field`, `Document`, `RecursiveCharacterTextSplitter`, `WebBaseLoader`, `FAISS`, `ChatGroq`, `HuggingFaceEmbeddings`, `StateGraph`, `END`. The `langchain-community is being sunset` warning is harmless.

### Cell 3: LLM and embeddings

```python
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

The original used `llama-3.3-70b-versatile`, which Groq no longer serves (`404 model_not_found` on the first call).

### Cell 4: Index two blog posts

```python
docs = WebBaseLoader(urls).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

Lilian Weng's *LLM Powered Autonomous Agents* and *Diffusion Models for Video Generation*, two deliberately unrelated topics in **one shared index**. `WebBaseLoader` accepts the list of URLs directly. **Output:** `Pages: 2 | Chunks indexed: 234`.

### Cell 5: State and plan schema

| `RAGState` field | Written by | Contents |
|---|---|---|
| `question` | Caller | The original question |
| `sub_questions` | `planner` | The planned sub-questions |
| `docs_per_subq` | `retriever` | `{sub-question: [new chunks]}` |
| `answer` | `responder` | Final answer |

`QueryPlan.sub_questions` has the description *"2-4 self-contained sub-questions. Each one must make sense on its own."* The LLM sees this description as part of the schema, so it acts as an instruction. `planner_llm = llm.with_structured_output(QueryPlan)`.

### Cell 6: Nodes
- **`plan_query`**: prompt with three rules (keep independent topics separate, do not invent connections, no numbering), then strips empty entries and prints the plan.
- **`retrieve_for_each`**: `retriever.batch(sub_questions)` returns one result list per sub-question. A shared `seen` set means a chunk is assigned only to the **first** sub-question that retrieved it, and later sub-questions get only new chunks.
- **`generate_final_answer`**: builds one `### Sub-question: ...` section per sub-question (or `(no new documents)`), then asks the LLM to answer using only that context, cover every sub-question, add a combined summary, and say so when something is not covered.

### Cell 7: Graph
`planner → retriever → responder → END`, with `set_entry_point("planner")`.

### Cell 8: Diagram
`draw_mermaid_png()` inside `try/except`, since it needs internet access to mermaid.ink.

### Cell 9: Run

```python
user_query = "Explain how agent loops work and what are the challenges in diffusion video generation?"
final_state = graph.invoke(RAGState(question=user_query))
```

`graph.invoke` returns a **dict**: read `final_state["answer"]`, not `final_state.answer`.

**Output of the saved run:**

```
🧩 Sub-questions:
  - How do agent loops work?
  - What are the challenges in diffusion video generation?
🔎 4 new docs for: How do agent loops work?
🔎 4 new docs for: What are the challenges in diffusion video generation?
```

The planner split the question cleanly into its two topics without inventing a link between them. Each sub-question got 4 chunks, with no overlap.

The answer has three parts:

- **Agent loops:** the LLM as the core controller plus memory, perception and action modules, planning, observations, and examples (AutoGPT, GPT-Engineer, BabyAGI, Generative Agents).
- **Diffusion video challenges:** temporal consistency across frames, the need for more world knowledge, scarce high-quality video and text-video data, heavier 3D U-Net / DiT architectures, and specialised sampling.
- **A combined summary.**

The agent-loop section is weaker than the video section. Some of its points (for example "learning history concatenation", which comes from the post's Algorithm Distillation section) are not really about the agent loop. The 4 chunks retrieved for that sub-question included loosely related passages, and the responder used them because it was told to use only the context. This is a retrieval-quality limit, not a planning error.

---

## 4. End-to-End Data Flow

```
 2 blog URLs ──► WebBaseLoader ──► splitter (500/50) ──► 234 chunks ──► MiniLM ──► FAISS (k=4)

 RAGState(question)
        │
        ▼
   planner ── structured output ──► sub_questions = [q1, q2]
        │
        ▼
   retriever ── retriever.batch([q1, q2]) ──► dedupe ──► docs_per_subq = {q1: [...], q2: [...]}
        │
        ▼
   responder ── "### Sub-question: q1 ... ### Sub-question: q2 ..." + question ──► gpt-oss-20b
        │
        ▼
       END ──► final_state["sub_questions"], final_state["answer"]
```

---

## 5. Limitations and Next Steps

| Limitation | Effect | Possible improvement |
|---|---|---|
| Plain top-k retrieval per sub-question | Loosely related chunks reach the answer (the agent-loop section above) | Add a reranker ([Reranking](../Reranking/readme.md)) or a relevance grader ([Agentic-RAG2](../Agentic-RAG2/readme.md)) |
| Dedupe assigns a shared chunk to the first sub-question only | A later sub-question can end up with few or no chunks | Keep shared chunks in every group, and dedupe only in the prompt |
| Always splits | Simple questions get split for no reason | Let the planner return a single sub-question when the question is simple |
| Sub-questions cannot depend on each other | "Find X, then use X to find Y" questions are not supported | Run the sub-questions in order and pass earlier answers to later ones |
| Index rebuilt on every run | Pages are downloaded and embedded each time | `vectorstore.save_local(...)` / `FAISS.load_local(...)` |
| No source citations | The reader cannot see which chunk supports each claim | Include `metadata["source"]` in each context section |

---

## 6. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `AssertionError: groq_api / GROQ_API_KEY missing in .env` | `.env` was not found, or has neither key. Run the notebook from inside the repository. |
| `404 model_not_found` for `llama-3.3-70b-versatile` | The model was retired on Groq. Use `openai/gpt-oss-20b`. |
| `USER_AGENT environment variable not set` warning | Cell 1 was not run before Cell 2. Restart the kernel and run from the top. |
| Sub-questions like "How do agent loops relate to diffusion video?" | The planner invented a connection. Keep the "Do NOT invent connections" rule in the prompt. |
| `AttributeError: 'dict' object has no attribute 'answer'` | `graph.invoke` returns a dict. Use `final_state["answer"]`. |
| Graph image does not render | `draw_mermaid_png()` needs internet. Use `print(graph.get_graph().draw_mermaid())` instead. |
