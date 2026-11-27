# Corrective RAG (CRAG) with LangGraph: Groq + Hugging Face + Tavily

This folder builds a **Corrective RAG** graph. Instead of trusting whatever the vector store returns, every retrieved chunk is **graded for relevance**. Based on the grades, the graph either answers straight away or **corrects** the retrieval by rewriting the query and searching the web.

| Grades | CRAG verdict | Action |
|---|---|---|
| All chunks relevant | **Correct** | Generate from the vector-store chunks |
| Some chunks relevant | **Ambiguous** | Keep the relevant chunks, add web results, then generate |
| No chunk relevant | **Incorrect** | Generate from web results only |

```
CorrectiveRAG/
├── code.ipynb     # Notebook: FAISS over 3 blog posts → grade → (rewrite + Tavily) → generate
└── readme.md      # This file
```

---

## 1. Core Concepts

### Why "corrective"?
Plain RAG has one weak point: if retrieval returns off-topic chunks, the LLM either answers from them anyway or says it does not know. Corrective RAG adds a **check after retrieval**:

```
retrieve ──► grade each chunk ──► good enough? ──yes──► generate
                                        │
                                        no
                                        ▼
                     rewrite query ──► web search ──► generate
```

A question outside the knowledge base (for example a sports result, when the vector store only holds AI blog posts) should end up being answered from the web instead of from unrelated chunks.

### The graph

```
START ──► retrieve ──► grade_documents ──decide_to_generate──► generate ──► END
                                │                                  ▲
                                └──► transform_query ──► web_search_node
```

| Node | Job |
|---|---|
| `retrieve` | FAISS search (`k=4`) with the user's question; tags each chunk `source_type="VECTORSTORE"` |
| `grade_documents` | Grades all chunks **in parallel** (`batch`), keeps only the `yes` chunks, prints the CRAG verdict and sets `web_search` |
| `decide_to_generate` | Conditional edge: `web_search == "Yes"` → `transform_query`, otherwise → `generate` |
| `transform_query` | Rewrites the question for web search into `search_query` |
| `web_search_node` | Tavily search (3 results) with `search_query`; adds them as `Document`s tagged `source_type="WEB"` |
| `generate` | Answers the **original** question from the tagged context |

### Two questions, kept apart
`GraphState` has separate fields for the user's question and the web query:

| Field | Written by | Used for |
|---|---|---|
| `question` | Caller, never changed | Retrieval, grading and the final answer |
| `search_query` | `transform_query` | Web search only |
| `documents` | `retrieve`, `grade_documents`, `web_search_node` | Context for `generate` |
| `web_search` | `grade_documents` | Routing (`"Yes"` / `"No"`) |
| `generation` | `generate` | The answer |

In the original CRAG tutorial the rewritten query **replaces** the question, so the final answer is written for the search-engine version of the question. Keeping both fields means the answer still matches what the user asked.

### Design choices

| Choice | Reason |
|---|---|
| `with_structured_output(GradeDocuments, method="json_schema")` | Groq's native JSON output. The default tool-calling mode sometimes fails on `gpt-oss-20b` with `400 tool_use_failed` (wrong tool name) |
| `binary_score: Literal["yes", "no"]` | The grader can only return one of two values, so routing never sees anything unexpected |
| `retrieval_grader.batch(...)` | All 4 chunks are graded at the same time instead of one after another |
| `format_docs` adds `[VECTORSTORE]` / `[WEB]` and the URL to each chunk | The LLM gets clean text with its source, not raw `Document` objects |
| `web_search_node` returns `state["documents"] + web_docs` | A new list instead of changing the state in place |
| `try/except` around Tavily | A failed web search does not stop the graph; it generates from what it has |
| `.strip().splitlines()[0]` on the rewritten query | Only the first line is used if the LLM adds extra text |

---

## 2. Setup

### API keys
`.env` in the repository root:

```env
groq_api=your_groq_api_key
hf_token=your_hugging_face_token
tavily_api=your_tavily_api_key
```

Cell 3 copies these into `GROQ_API_KEY`, `HF_TOKEN` and `TAVILY_API_KEY`. `TavilySearch` only reads `TAVILY_API_KEY`; without this step it fails with `ValidationError: Did not find tavily_api_key`.

### Packages
From the repository's `requirements.txt`: `langchain-groq`, `langchain-huggingface`, `sentence-transformers`, `langchain-community`, `langchain-tavily`, `faiss-cpu`, `langgraph`, `python-dotenv`. `WebBaseLoader` also needs `beautifulsoup4`.

---

## 3. Notebook Walkthrough

### Cell 1: Overview
Markdown summary of the five CRAG steps.

### Cell 2: Imports
`Document`, `HuggingFaceEmbeddings`, `TextLoader`, `FAISS`, `RecursiveCharacterTextSplitter`, `StateGraph`, `START`, `END`, `BaseModel`, `Field`. These replace the old `langchain.schema`, `langchain.vectorstores` and `OpenAIEmbeddings` imports.

### Cell 3: Environment
Loads `.env` with `find_dotenv(usecwd=True)`, sets the three keys and `USER_AGENT` (which removes the `WebBaseLoader` warning), and asserts the Groq and Tavily keys. **Output:** `Keys loaded: Groq True | Tavily True`.

### Cell 4: LLM and embeddings
`ChatGroq(model="openai/gpt-oss-20b", temperature=0)` and `all-MiniLM-L6-v2` embeddings. If grading gives errors, the comment suggests `openai/gpt-oss-120b`.

### Cell 5: Knowledge base
Loads three of Lilian Weng's blog posts with `WebBaseLoader`: *LLM Powered Autonomous Agents*, *Prompt Engineering* and *Adversarial Attacks on LLMs*. They are split into 1,500/100 character chunks (roughly 500 tokens, so `tiktoken` is not needed) and indexed in FAISS with `k=4`. **Output:** `Pages: 3 | Chunks: 116`.

### Cell 6: Retrieval grader
`GradeDocuments` (one field, `binary_score: "yes" | "no"`), a grading prompt and `retrieval_grader = grade_prompt | structured_llm_grader`. **Test:** question `"agent memory"`, second retrieved chunk → `binary_score='yes'`.

### Cell 7: RAG chain
The answer prompt: use **only** the context, mention the source tags used, say "I don't know" if the answer is missing. `format_docs` turns each chunk into `[SOURCE_TYPE] url` + text. **Test:** a table of short-term memory, long-term memory, memory stream and retrieval model, all tagged `[VECTORSTORE]`.

### Cell 8: Question rewriter
Rewrites a question for web search and returns only the new question. **Test:** `"agent memory"` → `"What is agent memory in artificial intelligence?"`.

### Cell 9: Web search tool
`TavilySearch(max_results=3)`.

### Cell 10: Graph state
`GraphState` with `question`, `search_query`, `generation`, `web_search` and `documents` (see the table above).

### Cell 11: Nodes
`retrieve`, `grade_documents`, `transform_query`, `web_search`, `generate` and the router `decide_to_generate`, as described in section 1. `grade_documents` prints one line per chunk and the verdict, for example `---CRAG: AMBIGUOUS (3/4 relevant) → keep good docs + web search---`.

### Cell 12: Build the graph

```python
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate,
                               {"transform_query": "transform_query", "generate": "generate"})
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()
```

The web search node is named `web_search_node` because `web_search` is already a state field.

### Cell 13: Diagram
`draw_mermaid_png()` inside `try/except` (needs internet access).

### Cell 14: Run

```python
result = app.invoke({"question": "What are the types of agent memory?"})
```

**Output of the saved run:**

| Step | Result |
|---|---|
| Grades | 3 of 4 chunks relevant → **AMBIGUOUS** |
| Search query | `"What are the different types of memory used by AI agents?"` |
| Documents used | `VECTORSTORE` × 3, `WEB` × 3 |
| Answer | A table of 8 memory types (in-context/working, episodic, semantic, procedural, organisational-context, retrieval/external, parametric, prospective), with the CoALA framework's four types named as the core list |

### Cell 15: Grader test with an off-topic question
Grades the 4 chunks retrieved for `"Who won the 2024 T20 Cricket World Cup?"`, a question the blog posts cannot answer.

**Output:** 3 chunks `NO`, but 1 chunk `YES`: a passage from the *Prompt Engineering* post about answer reranking scores (`PoE > Noisy channel > RAG`), which has nothing to do with cricket. See the next section.

---

## 4. What the Saved Outputs Show

### The grader is not reliable
In Cell 15 the grader marked a prompt-engineering chunk as relevant to a cricket question. In the full graph, this question would be classed as **AMBIGUOUS** instead of **INCORRECT**: the off-topic chunk would be kept and passed to `generate` together with the web results. The routing still reaches web search, so the answer can still be right, but an irrelevant chunk goes into the context. A likely cause is the lenient prompt: *"keyword(s) or semantic meaning that helps answer the question"* lets a chunk pass because of a loose link, such as the word "answer" or the general topic of question answering.

### Web results can take over the answer
In Cell 14 the vector-store chunks describe Lilian Weng's memory types (sensory, short-term, long-term), but the final answer is built mostly on the web results (CoALA, episodic/semantic/procedural memory). The answer also does not use the `[VECTORSTORE]` / `[WEB]` tags as the prompt asks; it ends with a general *"Sources: CoALA paper, AI-agent memory FAQ, and the 2026 technical guide"*. So it is hard to tell which claims came from the knowledge base.

---

## 5. End-to-End Data Flow

```
 {"question": ...}
        │
        ▼
 retrieve ── question ──► FAISS(116 chunks, k=4) ──► 4 docs [VECTORSTORE]
        │
        ▼
 grade_documents ── batch(4 × grader, json_schema) ──► keep "yes" docs
        │
        ├── 4/4 relevant (CORRECT) ──────────────────────────────────┐
        │                                                            │
        └── 1–3 relevant (AMBIGUOUS) or 0 (INCORRECT)                │
                 │                                                   │
                 ▼                                                   │
          transform_query ──► search_query                           │
                 │                                                   │
                 ▼                                                   │
          web_search_node ── Tavily(3) ──► + docs [WEB]              │
                 │                                                   │
                 ▼                                                   ▼
             generate ── question + tagged context ──► gpt-oss-20b ──► generation
```

---

## 6. Limitations and Next Steps

| Limitation | Effect | Possible improvement |
|---|---|---|
| Lenient grading prompt | Off-topic chunks pass (Cell 15) | Ask for a stricter check ("does it contain information that directly answers the question?"), or use a larger model such as `gpt-oss-120b` |
| Ambiguous and Incorrect take the same path | Both rewrite and search; the only difference is how many vector-store chunks are kept | Fewer web results for Ambiguous, more for Incorrect |
| Web results are not graded | Off-topic web content can reach the answer | Run the same grader on the web documents |
| Source tags are not enforced | Answers do not always cite `[VECTORSTORE]` / `[WEB]` | Require a tag after every claim and a `Sources used:` line, as in [AnswerSynthesis](../AnswerSynthesis/readme.md) |
| No check of the final answer | Unsupported claims are not caught | Add a groundedness check, as in [SelfReflection](../SelfReflection/readme.md) |
| Knowledge base is downloaded on every run | Slow, and fails without internet | Save the FAISS index with `vectorstore.save_local(...)` and load it |
| Original CRAG refines documents | The paper splits relevant documents into smaller strips and keeps only useful ones | Add a "knowledge refinement" step before `generate` |

---

## 7. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ValidationError: Did not find tavily_api_key` | `TAVILY_API_KEY` is not set. Cell 3 copies it from `tavily_api` in `.env`; restart the kernel and run from Cell 2. |
| `USER_AGENT environment variable not set` warning | Set `os.environ["USER_AGENT"]` before importing `langchain_community` (Cell 3 does this). |
| `400 tool_use_failed` from the grader | Tool-calling structured output on `gpt-oss-20b`. Keep `method="json_schema"`. |
| `⚠️ Web search failed: ...` | Tavily key, quota or network problem. The graph continues and generates from the remaining documents. |
| Every question goes to web search | The grader rejected at least one chunk. Inspect the grades as in Cell 15. |
| Off-topic questions are answered from the blog posts | The grader marked off-topic chunks as relevant. Use a stricter grading prompt (see section 6). |
| `WebBaseLoader` returns empty pages or fails | No internet, or the site blocked the request. Check the URLs in a browser. |
| Graph image does not render | `draw_mermaid_png()` needs internet. Use `print(app.get_graph().draw_mermaid())`. |
