# Chain-of-Thought RAG with LangGraph: Groq for Reasoning, Hugging Face for Retrieval

This folder builds a **Chain-of-Thought (CoT) RAG** pipeline. Instead of searching the vector store once with the whole question, an LLM first **breaks the question into smaller sub-questions**, each sub-question is searched separately, and all the retrieved context is combined into one reasoned answer.

The notebook is adapted from an OpenAI-based example. Every OpenAI component is replaced and the LangChain 1.x import paths are updated:

| Original | This notebook | Why |
|---|---|---|
| `init_chat_model("openai:gpt-4o")` | `init_chat_model("groq:openai/gpt-oss-20b", temperature=0)` | Groq-hosted open-weights model, same as the other modules |
| `OpenAIEmbeddings()` | `HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")` | Free, runs locally on the CPU |
| `from langchain.schema import Document` | `from langchain_core.documents import Document` | Old path removed in LangChain 1.x |
| `from langchain.vectorstores import FAISS` | `from langchain_community.vectorstores import FAISS` | Same |
| `from langchain.text_splitter import ...` | `from langchain_text_splitters import ...` | Same |
| `load_dotenv` imported but never called | `load_dotenv()` called | Without the call, `.env` is never read and the API key is empty |
| Plan parsed with `split("\n")` and `strip("- ")` | `llm.with_structured_output(Plan)` | Line parsing turns preambles like "Here are the steps:" and numbering like "1." into fake sub-questions |
| All retrieved chunks kept | Duplicates removed | Several sub-questions often return the same chunk |
| Sub-steps not passed to the answer prompt | Sub-steps included in the prompt | The answer can follow the plan |
| Nodes return `state.model_copy(update=...)` | Nodes return a partial dict | Only the changed field is returned; LangGraph merges it |

```
COT/
├── code.ipynb            # Notebook: FAISS index, planner → retriever → responder graph
├── research_notes.txt    # Fictional internal research notes (Experiments 14–16), the knowledge base
└── readme.md             # This file
```

---

## 1. Core Concepts

### What CoT RAG is
**Chain-of-Thought** prompting makes an LLM reason in intermediate steps instead of jumping to an answer. **CoT RAG** applies the same idea to retrieval:

```
User question
   ↓
planner    → split the question into 2–3 sub-questions      (Reason)
   ↓
retriever  → search the vector store once per sub-question  (Act)
   ↓
             merge and deduplicate the chunks              (Observe)
   ↓
responder  → reason through the steps, write the answer     (Reflect)
   ↓
  END
```

### Why split the question?
A single embedding of a broad question lands near only one part of the corpus. Searching with several narrower sub-questions reaches more of it:

| Single-query RAG | CoT RAG |
|---|---|
| One search with the full question | One search per sub-question |
| Finds the chunks closest to the question as a whole | Finds chunks for each part of the question |
| Can miss context for the second half of a multi-part question | Each part gets its own retrieval |
| 1 LLM call | 2 LLM calls (plan + answer) |

### How it differs from the other modules
| Module | Who decides what to retrieve |
|---|---|
| [Agentic-rag](../Agentic-rag/readme.md) | Nobody: always one search with the question |
| **COT (this module)** | The LLM plans the searches up front, then the graph runs them all |
| [React-Rag](../React-Rag/readme.md) | The LLM picks tools one at a time inside a loop |
| [Agentic-RAG2](../Agentic-RAG2/readme.md) | The LLM retrieves, a grader checks, and a rewrite loop retries |

The graph here is **linear**: no conditional edges, no loops. The only reasoning is in the planner and the final answer.

### Structured output for the plan
```python
class Plan(BaseModel):
    steps: List[str] = Field(description="2-3 short sub-questions, each searchable on its own")

planner_llm = llm.with_structured_output(Plan)
```
The LLM returns a `Plan` object, so `plan.steps` is already a clean Python list. Asking for **searchable** sub-questions matters: each step is sent to the retriever as-is.

---

## 2. Setup

### API keys
`.env` in the repository root:

```env
groq_api=your_groq_api_key
hf_token=your_hugging_face_token
```

| Key | Used for |
|---|---|
| `groq_api` (or `GROQ_API_KEY`) | The chat LLM (`openai/gpt-oss-20b` on Groq) |
| `hf_token` (or `HF_TOKEN`) | Downloading the embedding model from the Hugging Face Hub |

### Packages
All are in the repository's `requirements.txt`: `langchain`, `langchain-groq`, `langchain-huggingface`, `sentence-transformers`, `langchain-community`, `faiss-cpu`, `langgraph`, `python-dotenv`. `pydantic` and `langchain-text-splitters` come in as dependencies.

Run the notebook from inside `COT/` so `research_notes.txt` is found.

---

## 3. Notebook Walkthrough

### Cell 1: Imports
`init_chat_model`, `Document`, `HuggingFaceEmbeddings`, `TextLoader`, `FAISS`, `RecursiveCharacterTextSplitter`, `StateGraph`, `START`, `END`, plus `BaseModel` / `Field` for the state and the plan. The `langchain-community is being sunset` warning is harmless.

### Cell 2: Keys and the LLM

```python
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or os.getenv("groq_api", "")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") or os.getenv("hf_token", "")

llm = init_chat_model("groq:openai/gpt-oss-20b", temperature=0)
```

Displaying `llm` shows the `ChatGroq` object and its profile, including `structured_output: True`, which the planner relies on.

### Cell 3: Vector store

```python
docs = TextLoader("research_notes.txt", encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

**Output:** `5 chunks`. The test query `"transformer variants"` returns the Experiment 14 chunk first, then its conclusion line and the file's title line.

`k=3` per sub-question means up to 9 results across three steps. With only 5 chunks in the store, overlap between steps is guaranteed, which is why Cell 6 deduplicates.

### Cell 4: State

```python
class RAGCoTState(BaseModel):
    question: str
    sub_steps: List[str] = []
    retrieved_docs: List[Document] = []
    answer: str = ""
```

| Field | Written by |
|---|---|
| `question` | The caller |
| `sub_steps` | `planner` |
| `retrieved_docs` | `retriever` |
| `answer` | `responder` |

### Cell 5: Planner node
Sends the question with the instruction *"Break the question into 2-3 reasoning steps. Write each step as a short sub-question that can be searched on its own"* to the structured-output LLM, prints the plan and returns `{"sub_steps": plan.steps}`.

### Cell 6: Retriever node
Runs `retriever.invoke(sub)` for every sub-question and keeps each chunk only the first time its text is seen. Prints how many unique chunks were collected.

### Cell 7: Responder node
Builds one prompt with the original question, the plan as a bullet list and all retrieved chunks, and asks the LLM to *"work through the steps, then give a well-reasoned final answer using only the information above."*

### Cell 8: Graph

```python
builder.add_edge(START, "planner")
builder.add_edge("planner", "retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)
```

`add_edge(START, "planner")` does the same thing as the original `set_entry_point("planner")`.

### Cell 9: Run

```python
query = "What are the additional experiments in Transformer evaluation?"
final = graph.invoke(RAGCoTState(question=query))
```

`graph.invoke` returns a **dict**, so values are read as `final["answer"]`, not `final.answer`.

**Output of the saved run:**

- **Plan:** *"What are the standard evaluation experiments for Transformers?"*, *"What additional experiments are commonly performed in Transformer evaluation?"*, *"Which specific additional experiments are included in Transformer evaluation?"*
- **Retrieval:** 4 unique chunks out of 5, covering all three experiments.
- **Answer:** a table of Experiments 14, 15 and 16 with their key numbers (Longformer −45% memory, MoE +2.1 ROUGE-L, agent designs 61% / 74% / 82%, reflection −30% hallucinations / +1.8 s latency).

Two things to notice:

1. **The planner does not know what is in the corpus.** Its sub-questions are generic ("standard evaluation experiments for Transformers") rather than tied to the notes. Retrieval still worked because the file is tiny, but on a large corpus vague sub-questions retrieve vague context.
2. **The answer stretches the question.** Only Experiment 14 is about Transformers. Experiments 15 (agent designs) and 16 (reflection) were retrieved, and the model presented them as "additional Transformer experiments" too. The question is ambiguous, and the model fitted everything it was given to the question's wording.

---

## 4. End-to-End Data Flow

```
 research_notes.txt ──► TextLoader ──► splitter (500/50) ──► 5 chunks ──► MiniLM ──► FAISS (k=3)

 RAGCoTState(question)
        │
        ▼
   planner ── structured output ──► sub_steps = [q1, q2, q3]
        │
        ▼
   retriever ── retriever.invoke(q1), (q2), (q3) ──► dedupe ──► retrieved_docs (4 chunks)
        │
        ▼
   responder ── question + steps + context ──► gpt-oss-20b ──► answer
        │
        ▼
       END ──► final["sub_steps"], final["answer"]
```

---

## 5. Limitations and Next Steps

| Limitation | Effect | Possible improvement |
|---|---|---|
| Planning happens before any retrieval | Sub-questions are guesses, not based on what the corpus contains | Retrieve once with the full question first and give the planner that context |
| All steps are searched in one go | A later step cannot use what an earlier step found | Retrieve and reason step by step, passing each step's findings to the next |
| No relevance check | Off-topic chunks (Experiments 15–16 here) end up in the answer | Add a grading node before the responder, as in [Agentic-RAG2](../Agentic-RAG2/readme.md) |
| No source citations | The reader cannot see which chunk supports which claim | Number the chunks in the prompt and ask the model to cite them |
| Small corpus | With 5 chunks, almost everything is retrieved anyway | Try a larger document set to see the real benefit of splitting the question |

---

## 6. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `FileNotFoundError: research_notes.txt` | The notebook is not running from `COT/`. Open it from that folder or use a full path. |
| `ImportError` for `langchain.schema`, `langchain.vectorstores` or `langchain.text_splitter` | Old LangChain paths. Use the imports in Cell 1. |
| Groq authentication error | `groq_api` missing from `.env`, or `load_dotenv()` was not called. |
| Plan contains odd steps such as "Here are the steps:" | You are parsing plain text. Use `with_structured_output(Plan)` as in Cell 5. |
| `AttributeError: 'dict' object has no attribute 'answer'` | `graph.invoke` returns a dict. Use `final["answer"]`. |
