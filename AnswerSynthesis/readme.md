# Answer Synthesis from Multiple Sources with LangGraph: Groq + Hugging Face

This folder builds a RAG graph that gathers context from **four different sources at the same time** and **synthesizes one cited answer** from all of them:

| Tag | Source | How it is searched |
|---|---|---|
| `[INTERNAL]` | `internal_docs.txt`, a fictional engineering report | FAISS + Hugging Face embeddings |
| `[YOUTUBE]` | A mock video transcript | FAISS (one document) |
| `[WIKIPEDIA]` | Wikipedia summaries | `WikipediaAPIWrapper` |
| `[ARXIV]` | ArXiv paper abstracts | `ArxivAPIWrapper` |

A query transformer first turns the question into keywords for the public APIs. The four retrievers then run **in parallel**, and a synthesizer merges their results with a source tag after every claim.

```
AnswerSynthesis/
├── code.ipynb           # Notebook: 4 parallel retrievers → cited synthesis
├── internal_docs.txt    # Fictional engineering report on production transformer models (written by the notebook)
└── readme.md            # This file
```

---

## 1. Core Concepts

### Answer synthesis
Retrieval from one source gives one view. Real questions often need several:

| Source type | Strength | Weakness |
|---|---|---|
| Internal docs | Company-specific facts nobody else has | Narrow, not peer-reviewed |
| Video transcript | Plain-language explanations | Informal, may be shallow |
| Wikipedia | Broad, stable background | General, rarely recent |
| ArXiv | Recent research | Technical, keyword search can return off-topic papers |

**Answer synthesis** means combining these into one answer that says which source each claim came from, skips irrelevant sources, and keeps private notes apart from public research.

### The graph: fan-out and fan-in

```
                         ┌──► retrieve_text ───┐
                         ├──► retrieve_yt ─────┤
START ──► transform_query┤                     ├──► synthesize ──► END
                         ├──► retrieve_wiki ───┤
                         └──► retrieve_arxiv ──┘
```

- **Fan-out:** `transform_query` has four outgoing edges, so LangGraph runs all four retrievers in the **same step**, concurrently.
- **Fan-in:** `synthesize` has four incoming edges, so it waits until all four have finished.
- **No write conflicts:** each retriever returns only **its own** state field (`text_docs`, `yt_docs`, `wiki_context`, `arxiv_context`). If two parallel nodes wrote the same field, LangGraph would raise an `InvalidUpdateError` unless that field had a reducer.

### Two kinds of query
| Retriever | Query used | Why |
|---|---|---|
| Internal docs, YouTube (FAISS) | The full `question` | Embedding search works on meaning, so a natural sentence is best |
| Wikipedia, ArXiv (APIs) | `search_query` (3–8 keywords) | These are keyword search engines, and a long question returns poor matches |

### Making each source safe
| Protection | What it prevents |
|---|---|
| `MAX_CHARS_PER_SOURCE = 2500` + `_truncate` | One long source filling the prompt and exceeding Groq free-tier token limits |
| `try/except` in the Wikipedia and ArXiv nodes | A network or rate-limit error stopping the whole graph. The source is reported as `(unavailable)` |
| Fallback in `transform_query` | A failed keyword rewrite stopping the graph. The original question is used instead |
| `socket.setdefaulttimeout(30)` | A stuck API request hanging forever |
| Wikipedia user agent | Wikipedia blocking the request (see Cell 5) |

### Synthesizer rules
The synthesizer prompt labels each block (`[INTERNAL] Company internal engineering notes (private, NOT published research)`, `[YOUTUBE]`, `[WIKIPEDIA]`, `[ARXIV]`) and sets these rules:

1. Use only the context.
2. Cite a source tag after every claim.
3. Ignore sources that are not relevant.
4. Never present `[INTERNAL]` notes as public or published research.
5. Say so when sources disagree.
6. Do not invent connections between sources.
7. End with a `Sources used:` line.

---

## 2. Setup

### API keys
`.env` in the repository root:

```env
groq_api=your_groq_api_key
hf_token=your_hugging_face_token
```

Wikipedia and ArXiv need no keys.

### Packages
All are in the repository's `requirements.txt`: `langchain-groq`, `langchain-huggingface`, `sentence-transformers`, `langchain-community`, `faiss-cpu`, `langgraph`, `wikipedia`, `arxiv<2.2`, `python-dotenv`.

---

## 3. Notebook Walkthrough

### Cell 1: Overview
Markdown summary of the pipeline.

### Cell 2: Environment
Loads `.env` from the repository root, copies `groq_api` / `hf_token` into `GROQ_API_KEY` / `HF_TOKEN`, asserts the Groq key, and sets `USER_AGENT` before `langchain_community` is imported. **Output:** `GROQ_API_KEY loaded: True`.

### Cell 3: Imports
`Document`, `RecursiveCharacterTextSplitter`, `TextLoader`, `FAISS`, `WikipediaAPIWrapper`, `ArxivAPIWrapper`, `ChatGroq`, `HuggingFaceEmbeddings`, `StateGraph`, `END`, `BaseModel`, `Field`.

### Cell 4: LLM and embeddings
`ChatGroq(model="openai/gpt-oss-20b", temperature=0)` and one shared `all-MiniLM-L6-v2` embeddings object.

### Cell 5: Source helpers

```python
def load_text_retriever(file_path): ...     # 500/50 chunks → FAISS, k=3
def load_youtube_retriever(): ...           # one mock transcript Document → FAISS, k=1

import wikipedia
wikipedia.set_user_agent("RAG-AnswerSynthesis-Notebook/1.0 (learning project)")
socket.setdefaulttimeout(30)

wiki  = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2500)
arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=2500)
```

- The **YouTube source is mocked**: two sentences about agentic AI (feedback loops, memory, tool use, temporal reasoning). A real version would use `YoutubeLoader` / `youtube-transcript-api`.
- **The Wikipedia user agent is required.** Without it, Wikipedia blocks the request and returns an error page instead of JSON, so the wrapper fails with `Expecting value: line 1 column 1 (char 0)`.
- The API wrappers return summaries only (titles, authors and abstracts for ArXiv), not full pages or PDFs.

### Cell 6: Knowledge base
Writes `internal_docs.txt` (2,775 characters): transformer variants in production (EfficientFormer, DistilBERT, Reformer, LLaMA2, TinyBERT), deployment architecture, optimizations and additional notes (ViT/Swin, Wav2Vec 2.0, FlashAttention, RoPE). The same report is used in [IterativeRetrieval](../IterativeRetrieval/readme.md). The file is **overwritten on every run**.

### Cell 7: Build the FAISS retrievers
`text_retriever` and `youtube_retriever`.

### Cell 8: State and query schema

| `MultiSourceRAGState` field | Written by |
|---|---|
| `question` | Caller |
| `search_query` | `transform_query` |
| `text_docs` | `retrieve_text` |
| `yt_docs` | `retrieve_yt` |
| `wiki_context` | `retrieve_wiki` |
| `arxiv_context` | `retrieve_arxiv` |
| `final_answer` | `synthesize` |

```python
query_transformer = llm.with_structured_output(SearchQuery, method="json_schema")
```

`method="json_schema"` uses Groq's native structured output. The default mode goes through a tool call, and `gpt-oss-20b` sometimes names that tool wrongly (`searchQuery` instead of `SearchQuery`), which Groq rejects with `400 tool_use_failed`.

### Cell 9: Nodes
- **`transform_query`**: asks for a short keyword query. Falls back to the original question if that fails.
- **`retrieve_text` / `retrieve_yt`**: FAISS search with the full question.
- **`retrieve_wikipedia` / `retrieve_arxiv`**: keyword search, truncated, and wrapped in `try/except`.
- **`synthesize_answer`**: joins each source under its tag (or `(nothing found)`), applies the rules above and returns `final_answer`.

### Cell 10: Graph

```python
builder.set_entry_point("transform_query")
for node in ["retrieve_text", "retrieve_yt", "retrieve_wiki", "retrieve_arxiv"]:
    builder.add_edge("transform_query", node)   # fan-out
    builder.add_edge(node, "synthesize")         # fan-in
builder.add_edge("synthesize", END)
```

### Cell 11: Diagram
`draw_mermaid_png()` inside `try/except` (needs internet access).

### Cell 12: Run

```python
question = "What are transformer agents and how are they evolving in recent research?"
result = graph.invoke(MultiSourceRAGState(question=question))
```

**Output of the saved run:** search query `'transformer agents recent research evolution'`. The answer has three cited sections:

| Section | Source |
|---|---|
| Efficient transformer variants in production (EfficientFormer, ViT/Swin, Wav2Vec 2.0, FlashAttention, RoPE) | `[INTERNAL]` |
| Agentic features: feedback loops, memory, tool use, temporal reasoning | `[YOUTUBE]` |
| Multi-agent coordination through communication | `[ARXIV]` |

It ends with `Sources used: [INTERNAL], [YOUTUBE], [ARXIV]`. Wikipedia was ignored, correctly, because it only returned a general *Artificial intelligence* page.

### Cell 13: Inspect each source
Prints the search query, the number of internal and YouTube docs (3 and 1), and the first 500 characters of the Wikipedia and ArXiv context. This is the quickest way to see **why** the answer looks the way it does.

It shows that ArXiv's first result was *"Fixation probability of rare nonmutator and evolution of mutation rates"*, a 2015 biology paper. The word "evolution" in the keyword query matched evolutionary biology. The synthesizer ignored it, so the relevance rule worked, but the example shows how keyword queries can go wrong.

### What to watch for in the answer
The synthesizer follows the citation rules well, but it still **frames sources to fit the question**. Section 1 presents internal production models such as EfficientFormer as part of how "transformer agents" are evolving, but the internal report never mentions agents. The facts are cited correctly while the link between them is the model's own. In an earlier run without the "do not invent connections" rule, this was stronger: uncited claims and "maintaining agentic performance" attributed to the internal notes. Prompt rules reduce this but do not remove it.

---

## 4. End-to-End Data Flow

```
 MultiSourceRAGState(question)
        │
        ▼
 transform_query ── json_schema ──► search_query (keywords)
        │
        ├──► retrieve_text  ── question ──► FAISS(internal_docs, k=3) ──► text_docs
        ├──► retrieve_yt    ── question ──► FAISS(mock transcript, k=1) ──► yt_docs
        ├──► retrieve_wiki  ── keywords ──► Wikipedia (2 pages, ≤2500 chars) ──► wiki_context
        └──► retrieve_arxiv ── keywords ──► ArXiv (2 abstracts, ≤2500 chars) ──► arxiv_context
                       (all four in parallel)
        │
        ▼
 synthesize ── [INTERNAL] [YOUTUBE] [WIKIPEDIA] [ARXIV] blocks + rules ──► gpt-oss-20b
        │
        ▼
       END ──► result["final_answer"]
```

---

## 5. Limitations and Next Steps

| Limitation | Effect | Possible improvement |
|---|---|---|
| Keyword queries are ambiguous | ArXiv returned a biology paper for "evolution" | Add domain words to the query (e.g. "LLM", "machine learning"), or use separate queries per source |
| No relevance filtering before synthesis | Off-topic results still use prompt space | Grade each source before the synthesizer, as in [Agentic-RAG2](../Agentic-RAG2/readme.md) |
| Synthesizer links sources on its own | Cited facts are joined by uncited reasoning | Add a grounding check afterwards, as in [SelfReflection](../SelfReflection/readme.md) |
| Mock YouTube source | Two sentences, so claims based on it are thin | Load a real transcript with `YoutubeLoader` |
| Same LLM for every step | No independent check | Use a different model for synthesis or checking |
| Knowledge base rewritten every run | Edits to `internal_docs.txt` are lost | Write the file only when it does not exist |

---

## 6. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `400 tool_use_failed ... attempted to call tool 'searchQuery'` | Tool-calling structured output on `gpt-oss-20b`. Use `with_structured_output(SearchQuery, method="json_schema")`. |
| `⚠️ Wikipedia failed: Expecting value: line 1 column 1 (char 0)` | Wikipedia blocked the request. Call `wikipedia.set_user_agent(...)` before searching (Cell 5). |
| `⚠️ ArXiv failed: ...` / HTTP 429 | ArXiv rate limit. Wait and rerun; the graph continues without it. |
| `InvalidUpdateError` after adding a node | Two parallel nodes write the same state field. Give each node its own field, or add a reducer. |
| Answer is mostly from one source | The others returned nothing relevant. Run Cell 13 to inspect each source. |
| Graph image does not render | `draw_mermaid_png()` needs internet. Use `print(graph.get_graph().draw_mermaid())`. |
