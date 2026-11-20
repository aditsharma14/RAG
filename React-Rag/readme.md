# ReAct RAG with LangGraph: One Agent, Many Knowledge Sources

This folder combines the two previous modules. [ReAct-Agents](../ReAct-Agents/readme.md) showed an LLM calling tools in a loop, and [Agentic-rag](../Agentic-rag/readme.md) showed RAG as a LangGraph graph. Here, **retrieval itself becomes a set of tools**: a web blog post, two private text files, Wikipedia and ArXiv are each wrapped as a separate tool. A ReAct agent driven by a Groq LLM then decides which sources to search for each question, and how many times.

The notebook builds this in two stages:

1. **One private source + Wikipedia**: a FAISS retriever over Lilian Weng's agents blog post, plus a Wikipedia tool.
2. **Several private sources + public research**: a helper that turns any text file into a retriever tool, two internal "company" documents, Wikipedia and ArXiv, with a system prompt that asks the agent to cite its sources.

```
React-Rag/
├── code.ipynb           # Notebook: retriever tools, Wikipedia/ArXiv tools, ReAct agent inside a LangGraph graph
├── internal_docs.txt    # Fictional internal tech docs ("Nimbus" RAG platform at Acme AI)
├── research_notes.txt   # Fictional internal research notes (transformer and agent design experiments)
└── readme.md            # This file
```

---

## 1. Core Concepts

### ReAct RAG
**ReAct = Reason + Act.** Instead of retrieving once and answering, the LLM works in a loop:

```
Question ──► Reason: which source do I need?
                │
                ▼
             Act: call a tool (InternalTechDocs, Wikipedia, ...)
                │
                ▼
             Observe: read the tool output
                │
       enough information? ──no──► reason again (same tool or another)
                │
               yes
                ▼
           Final answer
```

| Plain RAG (Agentic-rag module) | ReAct RAG (this module) |
|---|---|
| Always retrieves, from one index | Retrieves only when the model decides it is needed |
| One retrieval per question | Can search several times with different queries |
| One knowledge source | Picks between private docs, research notes, Wikipedia and ArXiv |
| Fixed graph: `retriever → responder` | The loop lives inside the agent node |

### One tool per knowledge source
Every source gets its **own tool with its own description**, instead of one big index holding everything. The LLM only sees each tool's name and description, so a clear description such as *"Search internal research notes for experimental results and agent designs"* is what makes the model pick the right source.

The included `research_notes.txt` even "measures" this idea (Experiment 15): one retriever tool scored 74%, separate tools per source scored 82%. The numbers are made up for the demo, but the design principle is real.

### `create_agent` as a graph node
`create_agent(llm, tools)` (the LangChain 1.x successor of `create_react_agent`) returns a **compiled LangGraph graph** that already contains the full ReAct loop: LLM node, tool node and the conditional edge between them. A compiled graph can be added to another graph as a single node, so the outer graph here is simply:

```
START ──► react_agent ──► END
```

The outer graph adds nothing yet, but it is the place to add more steps later, such as a query rewriter before the agent or an answer checker after it.

### `Tool` vs `@tool`
This notebook uses the `Tool(name=..., description=..., func=...)` class instead of the `@tool` decorator. `Tool` wraps a function that takes **one string**, so the model sends its query as a single argument called `__arg1` (visible in the tool-call printouts). That is all a search tool needs.

---

## 2. Setup

### API keys
Create a `.env` file in the repository root:

```env
groq_api=your_groq_api_key
hf_token=your_hugging_face_token
```

| Key | Used for |
|---|---|
| `groq_api` (or `GROQ_API_KEY`) | The chat LLM (`openai/gpt-oss-20b` on Groq) |
| `hf_token` (or `HF_TOKEN`) | Downloading the embedding model from the Hugging Face Hub (public model, the token mainly avoids rate limits) |

Wikipedia and ArXiv need no keys.

### Packages
All of these are in the repository's `requirements.txt`:

| Package | Why |
|---|---|
| `langchain`, `langchain-groq` | `init_chat_model`, `create_agent`, the Groq backend |
| `langchain-huggingface`, `sentence-transformers` | Local `all-MiniLM-L6-v2` embeddings |
| `langchain-community`, `faiss-cpu` | `FAISS`, `WebBaseLoader`, `TextLoader`, the Wikipedia/ArXiv wrappers |
| `langgraph` | `StateGraph`, `add_messages` |
| `wikipedia` | Used by `WikipediaAPIWrapper` |
| `arxiv<2.2` | Used by `ArxivAPIWrapper` |
| `python-dotenv` | Loading `.env` |

Run the notebook from inside `React-Rag/` so that `internal_docs.txt` and `research_notes.txt` are found by their relative paths.

---

## 3. Notebook Walkthrough

### Cell 1: Imports

| Import | Purpose |
|---|---|
| `init_chat_model` | Create the chat model from a `"provider:model"` string |
| `create_agent` | Prebuilt ReAct agent (replaces `create_react_agent`) |
| `Tool` | Wrap a plain function as a tool (moved to `langchain_core.tools`) |
| `BaseMessage`, `HumanMessage` | Message types for the graph state and the user query |
| `HuggingFaceEmbeddings` | Local embedding model (replaces `OpenAIEmbeddings`) |
| `FAISS` | In-memory vector index |
| `WebBaseLoader`, `TextLoader` | Load a web page / a local text file as `Document`s |
| `WikipediaAPIWrapper`, `ArxivAPIWrapper` | Search Wikipedia and ArXiv |
| `RecursiveCharacterTextSplitter` | Split text into overlapping chunks |
| `StateGraph`, `START`, `END`, `add_messages` | Build the outer graph |

The `langchain-community is being sunset` and `USER_AGENT environment variable not set` warnings are harmless.

### Cell 2: Keys, network settings and the LLM

```python
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or os.getenv("groq_api", "")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") or os.getenv("hf_token", "")
os.environ["USER_AGENT"] = "react-rag-notebook"

wikipedia.set_user_agent("RAG-ReAct-Notebook/1.0 (learning project)")
socket.setdefaulttimeout(30)

llm = init_chat_model("groq:openai/gpt-oss-20b", temperature=0)
```

- Keys are accepted under either the uppercase or the lowercase name.
- **Wikipedia user agent**: without one, Wikipedia often answers with HTTP 429 (too many requests).
- **Socket timeout**: the `wikipedia` and `arxiv` libraries set no timeout of their own, so a stuck request would hang the cell forever. 30 seconds is a global limit.
- `temperature=0` makes the tool choices and answers as repeatable as possible.

### Cell 3: Index the blog post

```python
docs = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()
```

Downloads *LLM Powered Autonomous Agents*, splits it into 500-character chunks with 50 characters of overlap, embeds them locally and stores them in FAISS. The test query `"what are autonomous agents"` returns the 4 closest chunks.

### Cell 4: The retriever as a tool

```python
retriever_tool = Tool(
    name="RAGRetriever",
    description="Use this tool to fetch relevant knowledge base info about LLM-powered autonomous agents (planning, memory, tool use).",
    func=retriever_tool_func,
)
```

`retriever_tool_func` runs the retriever and joins the chunk texts into one string, since a tool must return text the LLM can read. The `print("📚 Using RAGRetriever tool")` line makes every call visible in the cell output.

### Cell 5: Wikipedia tool

```python
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500)
```

Returns the summary of the single best-matching page, cut to 1,500 characters to keep the prompt small. Any exception is caught and returned as text (`"Wikipedia search failed ... Answer from other sources."`), so a network error does not crash the agent. The model reads the message and carries on with its other tools.

### Cells 6–7: The agent and the outer graph

```python
react_node = create_agent(llm, [retriever_tool, wiki_tool])

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

builder = StateGraph(AgentState)
builder.add_node("react_agent", react_node)
builder.add_edge(START, "react_agent")
builder.add_edge("react_agent", END)
graph = builder.compile()
```

The state has one key, `messages`. The `add_messages` reducer **appends** new messages instead of replacing the list, so the full history (question, tool calls, tool outputs, answer) builds up as the agent works.

### Cells 8–9: First run and inspecting the messages

```python
user_query = "What is an agent loop and how does Wikipedia describe autonomous agents?"
result = graph.invoke({"messages": [HumanMessage(content=user_query)]})
```

**Output:** the agent called **Wikipedia six times** with different queries and never called `RAGRetriever`, even though the blog post covers agent loops. The question says "Wikipedia" explicitly, and the model followed that wording. This is a useful lesson: the agent's tool choice depends on the question's wording and the tool descriptions, and it can skip a better source. The final answer is a table of the loop's stages (perception, memory, planning, action, feedback).

Cell 9 calls `m.pretty_print()` on every message to show the trace: `Human Message → Ai Message (tool call) → Tool Message → ... → Ai Message (final answer)`.

### Cell 10: A factory for file-based retriever tools

```python
def make_retriever_tool_from_text(file, name, desc):
    docs = TextLoader(file, encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vs = FAISS.from_documents(chunks, embedding)
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    ...
    return Tool(name=name, description=desc, func=tool_func)
```

Each call builds a **separate FAISS index** for one file and wraps it in a tool. It reuses the embedding model from Cell 3 and returns the top 3 chunks, separated by blank lines.

### Cell 11: ArXiv tool

```python
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
```

Returns the title, authors and abstract of the 2 best-matching papers. ArXiv rate-limits rapid calls with HTTP 429, so errors are returned as text, the same way as the Wikipedia tool.

### Cell 12: The two internal tools

| Tool | File | Contents |
|---|---|---|
| `InternalTechDocs` | `internal_docs.txt` | Architecture of the fictional "Nimbus" RAG stack: ingestion, hybrid retrieval (FAISS + BM25 + RRF), cross-encoder reranking, a 0.35 confidence cut-off, 8-step agent cap, Redis caching |
| `InternalResearchNotes` | `research_notes.txt` | Experiments 14–16: Longformer vs MoE transformers, three agent designs, a self-reflection step |

The test query `"transformer variants"` returns Experiment 14.

Both files are invented so that the answers **cannot come from the LLM's training data**. If the agent mentions "Nimbus" or the 82% figure, it must have retrieved them.

### Cell 13: The multi-source agent

```python
tools = [wiki_tool, arxiv_tool, internal_tool_1, internal_tool_2]
react_node = create_agent(
    llm,
    tools,
    system_prompt="You are a research assistant. Use the internal tools for company-specific questions "
                  "and Wikipedia/ArXiv for public knowledge. Say which source each fact came from.",
)
```

The system prompt gives a routing rule (internal tools for company questions, public tools for general knowledge) and asks for a source on every fact. The graph is rebuilt the same way as in Cell 7, with the node renamed `agentic_rag`.

### Cell 14: Internal + public question

> *"What do our internal research notes say about transformer variants, and what does ArXiv suggest recently?"*

**Tools called:** `InternalResearchNotes` once, ArXiv twice.
**Output:** one table of the Longformer/MoE findings from the notes and one table of recent ArXiv papers, each labelled with its source.

### Cell 15: Two internal sources in one question

> *"How does our Nimbus system do retrieval, and which agent design worked best in our experiments?"*

```python
for m in result["messages"]:
    for tc in getattr(m, "tool_calls", []) or []:
        print("🔧", tc["name"], "->", tc["args"])
```

**Tools called:** `InternalTechDocs` with `"Nimbus retrieval mechanism"` and `InternalResearchNotes` with `"Nimbus agent design experiments"`. The agent split the question in two and sent each half to the right source.
**Output:** the hybrid retrieval pipeline (FAISS + BM25 → RRF → cross-encoder → top 5), followed by the agent design results with the multi-tool ReAct design (82%) as the winner.

---

## 4. End-to-End Data Flow

```
            ┌──────────────── indexing (runs once) ────────────────┐
 blog URL ──► WebBaseLoader ─┐
 internal_docs.txt ──► TextLoader ─┼─► splitter (500/50) ─► all-MiniLM-L6-v2 ─► one FAISS index per source
 research_notes.txt ─► TextLoader ─┘
            └───────────────────────────────────────────────────────┘

 {"messages": [HumanMessage]}
        │
        ▼
 START ──► agentic_rag node (create_agent)
             │
             │   ┌──────────── ReAct loop ────────────┐
             │   │ Groq gpt-oss-20b                   │
             │   │   │ tool_calls?                    │
             │   │   ├─yes─► InternalTechDocs         │
             │   │   │       InternalResearchNotes    │
             │   │   │       Wikipedia / ArxivSearch  │
             │   │   │        │ ToolMessage           │
             │   │   ◄────────┘                       │
             │   │   no → final AIMessage             │
             │   └────────────────────────────────────┘
             ▼
            END ──► result["messages"][-1].content
```

---

## 5. Limitations and Next Steps

| Limitation | Effect | Possible improvement |
|---|---|---|
| Tool choice follows the question's wording | Cell 8 skipped the blog retriever because the question said "Wikipedia" | Sharper tool descriptions, or a system prompt that tells the agent to check private sources first |
| No step limit | A confused agent could loop for a long time (cell 8 made 6 Wikipedia calls) | Pass `config={"recursion_limit": N}` to `graph.invoke` |
| No memory | Every `invoke` starts fresh | Compile with `checkpointer=MemorySaver()` and pass a `thread_id` |
| No answer check | Nothing verifies the answer against the retrieved text | Add a reflection node after the agent (Experiment 16 in the notes) |
| Sources named, not linked | Answers say "InternalTechDocs" but not which chunk | Return `doc.metadata` along with the text in each tool |
| Indexes rebuilt every run | Re-downloading and re-embedding each time | `vectorstore.save_local(...)` / `FAISS.load_local(...)` |

---

## 6. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `FileNotFoundError: internal_docs.txt` | The notebook's working directory is not `React-Rag/`. Open the notebook from that folder or use full paths. |
| `Wikipedia search failed (...)` or `ArXiv search failed (...)` in a tool output | Rate limiting (HTTP 429) or a network timeout. The agent continues with its other tools; rerun the cell later. |
| Cell hangs on a Wikipedia/ArXiv call | Make sure Cell 2 ran, since it sets `socket.setdefaulttimeout(30)`. |
| `ImportError: cannot import name 'create_react_agent'` / `Tool` from `langchain.agents` | LangChain 1.x moved them. Use `from langchain.agents import create_agent` and `from langchain_core.tools import Tool`. |
| Groq authentication error | `groq_api` is missing from `.env`, or `load_dotenv()` could not find the file. |
| Agent ignores a tool you expected it to use | Check the tool's description and the question's wording; the LLM chooses tools from those alone. |
