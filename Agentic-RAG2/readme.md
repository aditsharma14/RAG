# Agentic RAG with Document Grading and Query Rewriting (Groq + Hugging Face)

This folder builds a **self-correcting agentic RAG** graph in LangGraph. Earlier modules either always retrieved ([Agentic-rag](../Agentic-rag/readme.md)) or let an agent pick tools without checking the results ([React-Rag](../React-Rag/readme.md)). This graph adds a **quality check**: after retrieval, an LLM grader decides whether the documents are actually relevant. If they are not, the question is rewritten and the agent tries again.

The notebook is adapted from an OpenAI-based example. Every OpenAI component is replaced, and some APIs removed in LangChain 1.x are updated:

| Original | This notebook | Why |
|---|---|---|
| `OpenAIEmbeddings()` | `HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")` | Free, runs locally on the CPU |
| `ChatGroq(model="qwen-qwq-32b")` | `init_chat_model("groq:openai/gpt-oss-20b", temperature=0)` | `qwen-qwq-32b` was retired by Groq |
| A new `ChatGroq(...)` inside every node | One `llm`, created once and shared | No reason to rebuild the client on every call |
| `from langchain.tools.retriever import create_retriever_tool` | `from langchain_core.tools import create_retriever_tool` | Old path removed in LangChain 1.x |
| `hub.pull("rlm/rag-prompt")` | The same prompt written inline | `langchain.hub` is not in LangChain 1.x, and no network call is needed |
| `rewrite` returns an `AIMessage` | `rewrite` returns a `HumanMessage` | The improved question should reach the agent as a new user question |
| `binary_score: str` | `binary_score: Literal["yes", "no"]` | Stops the grader from returning `"Yes."` or a full sentence |
| Vector-store code copied twice | `build_retriever(urls)` helper | Same steps, written once |
| No loop limit | `recursion_limit: 15` | Stops an endless rewrite → agent → retrieve cycle |

```
Agentic-RAG2/
├── code.ipynb   # Notebook: two FAISS retriever tools, agent / grade / rewrite / generate graph
└── readme.md    # This file
```

---

## 1. Core Concepts

### The graph

```
START ──► agent ──tool call?──no──► END        (answers directly)
            ▲         │
            │        yes
            │         ▼
         rewrite ◄── retrieve ──► grade_documents
         (not relevant)                │ (relevant)
                                       ▼
                                   generate ──► END
```

| Node / edge | Type | Job |
|---|---|---|
| `agent` | Node | The LLM, with both retriever tools bound, decides whether to retrieve or answer directly |
| `tools_condition` | Conditional edge | Routes to `retrieve` if the last message has `tool_calls`, otherwise to `END` |
| `retrieve` | Node (`ToolNode`) | Runs the tool the LLM asked for and adds a `ToolMessage` with the retrieved text |
| `grade_documents` | Conditional edge | An LLM with structured output answers `yes`/`no`: are these documents relevant to the question? |
| `generate` | Node | Answers the question from the retrieved context using a RAG prompt |
| `rewrite` | Node | Asks the LLM for a clearer version of the question and sends it back to `agent` |

The `rewrite → agent` edge is what makes the graph **self-correcting**. A plain RAG pipeline answers from whatever it retrieves, even bad context. Here, bad context is caught before it reaches the answer.

### Two knowledge sources, two tools
Each documentation set gets its own FAISS index and its own tool:

| Tool | Source | Description the LLM sees |
|---|---|---|
| `retriever_vector_db_blog` | LangGraph docs (3 pages) | "Search and return information about LangGraph" |
| `retriever_vector_langchain_blog` | LangChain docs (3 pages) | "Search and return information about LangChain" |

The LLM picks a tool only from its name and description, so these sentences act as routing rules.

### Grading with structured output
`llm.with_structured_output(Grade)` makes the LLM return a `Grade` object instead of free text. Because `binary_score` is typed `Literal["yes", "no"]`, the result can be compared directly in an `if`, with no string parsing.

### State
```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```
The whole run is one growing message list. Each node reads from it by position:

- `messages[0]`: the user's original question (used by grader, generator and rewriter)
- `messages[-1]`: the latest message, which is the `ToolMessage` holding retrieved text when the grader and generator run

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
| `groq_api` | The chat LLM (`openai/gpt-oss-20b` on Groq) |
| `hf_token` | Downloading the embedding model from the Hugging Face Hub |

### Packages
All are in the repository's `requirements.txt`: `langchain`, `langchain-groq`, `langchain-huggingface`, `sentence-transformers`, `langchain-community`, `faiss-cpu`, `langgraph`, `python-dotenv`. `beautifulsoup4`, `langchain-text-splitters` and `pydantic` come in as dependencies.

---

## 3. Notebook Walkthrough

### Cell 1: Environment
Loads `.env`, copies `groq_api` / `hf_token` into `GROQ_API_KEY` / `HF_TOKEN`, and sets `USER_AGENT` for `WebBaseLoader`.

### Cell 2: Imports, LLM and embeddings

```python
llm = init_chat_model("groq:openai/gpt-oss-20b", temperature=0)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

One LLM serves all four roles: agent, grader, generator and rewriter. `temperature=0` keeps routing decisions repeatable. The embedding model is downloaded on first use (the `Loading weights` bar) and cached after that.

### Cell 3: Build the two retrievers

```python
def build_retriever(urls):
    docs = [d for url in urls for d in WebBaseLoader(url).load()]
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    return FAISS.from_documents(splits, embedding).as_retriever()
```

Load each URL, flatten the results into one list, split into 1,000-character chunks with 100 characters of overlap, embed and index in FAISS. Called once for the LangGraph URLs and once for the LangChain URLs.

**Output, and the main issue in this run:** every chunk returned by `retriever.invoke("what is langgraph")` is just `"Redirecting..."`. The `langchain-ai.github.io/langgraph/...` and `python.langchain.com/docs/...` pages have moved to `docs.langchain.com`, and the old URLs now serve a small redirect page. `WebBaseLoader` does not follow this kind of client-side redirect, so it indexes the redirect stub instead of the docs. See [Section 5](#5-known-issue-the-source-urls) for the fix.

### Cell 4: Retriever tools

```python
retriever_tool = create_retriever_tool(retriever, "retriever_vector_db_blog", "Search and return information about LangGraph")
retriever_tool_langchain = create_retriever_tool(retriever_langchain, "retriever_vector_langchain_blog", "Search and return information about LangChain")
tools = [retriever_tool, retriever_tool_langchain]
llm_with_tools = llm.bind_tools(tools)
```

`create_retriever_tool` turns a retriever into a tool that takes a `query` string and returns the matching chunks joined as text. `bind_tools` is called once, and the bound model is reused on every `agent` call.

### Cell 5: State
`AgentState` with the `add_messages` reducer, so node outputs are appended to the message list rather than replacing it.

### Cell 6: Nodes and the grading edge

- **`agent`**: `llm_with_tools.invoke(messages)`. Returns either a tool call or a direct answer.
- **`grade_documents`**: fills the grader prompt with `messages[0]` (question) and `messages[-1]` (retrieved text) and returns `"generate"` or `"rewrite"`. The return type `Literal["generate", "rewrite"]` also tells LangGraph which nodes this edge can lead to, which is why `add_conditional_edges("retrieve", grade_documents)` needs no mapping dict.
- **`generate`**: the `rlm/rag-prompt` text (answer from context, say "I don't know" if unsure, three sentences max) piped into the LLM and a `StrOutputParser`.
- **`rewrite`**: asks the LLM to reason about the intent of the question and reply with only an improved version, then returns it as a `HumanMessage`.

### Cell 7: Build and draw the graph

```python
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")
```

`draw_mermaid_png()` renders the graph diagram. It uses the mermaid.ink web service, so it needs internet access.

### Cell 8: Run three questions

```python
cfg = {"recursion_limit": 15}
for q in ["What is Langgraph?", "What is Langchain?", "What is Machine learning?"]:
    result = graph.invoke({"messages": [HumanMessage(content=q)]}, config=cfg)
```

What happened in the saved run:

| Question | Path taken | Result |
|---|---|---|
| What is Langgraph? | agent → retrieve → grade (**not relevant**) → rewrite → agent → END | The grader correctly rejected the `"Redirecting..."` chunks. After the rewrite, the agent answered from its own knowledge without retrieving again |
| What is Langchain? | agent → END | No tool call; answered from the model's own knowledge |
| What is Machine learning? | agent → END | No tool call, which is expected: neither tool is about general ML |

So the **grading and rewrite path works**: it stopped useless context from reaching the answer. But because the indexes are empty, **no answer in this run actually came from the documents**.

---

## 4. End-to-End Data Flow

```
 ┌──────────────────── indexing (runs once) ────────────────────┐
 3 LangGraph URLs  ──► WebBaseLoader ──► splitter (1000/100) ──► MiniLM ──► FAISS ──► retriever_vector_db_blog
 3 LangChain URLs  ──► WebBaseLoader ──► splitter (1000/100) ──► MiniLM ──► FAISS ──► retriever_vector_langchain_blog
 └───────────────────────────────────────────────────────────────┘

 {"messages": [HumanMessage(question)]}
        │
        ▼
     agent (gpt-oss-20b + tools) ──no tool call──► END
        │ tool call
        ▼
     retrieve (ToolNode) ── ToolMessage(chunks)
        │
        ▼
     grade_documents (structured yes/no)
        ├── yes ──► generate (RAG prompt) ──► END
        └── no  ──► rewrite ──► HumanMessage(better question) ──► agent
```

---

## 5. Known Issue: the Source URLs

The original example's URLs no longer return documentation. To get real content into the indexes, replace them with the current pages on `docs.langchain.com` (open each old URL in a browser to see where it redirects), then check before building the graph:

```python
retriever.invoke("what is langgraph")[0].page_content[:300]
```

If the output still says `Redirecting...`, the URL is still a redirect stub.

---

## 6. Limitations and Next Steps

| Limitation | Effect | Possible improvement |
|---|---|---|
| Agent may skip retrieval | "What is Langchain?" was answered without the LangChain tool | Add a system prompt telling the agent to always check the docs for LangChain/LangGraph questions |
| Grader sees all chunks as one block | One relevant chunk among irrelevant ones can still pass or fail as a whole | Grade each chunk and keep only the relevant ones |
| Rewrite always reads `messages[0]` | A second rewrite starts from the original question again, not the first rewrite | Read the latest `HumanMessage` instead |
| No memory | Every `invoke` starts fresh | Compile with `checkpointer=MemorySaver()` and pass a `thread_id` |
| Indexes rebuilt on every run | Pages are downloaded and embedded each time | `save_local` / `FAISS.load_local` |

---

## 7. Troubleshooting

| Problem | Cause and fix |
|---|---|
| Retrieved chunks are `Redirecting...` | Old documentation URLs. See [Section 5](#5-known-issue-the-source-urls). |
| `GraphRecursionError` | The rewrite loop hit `recursion_limit` because retrieval kept failing the grade. Fix the sources, or raise the limit. |
| `ImportError` for `langchain.tools.retriever` or `langchain.hub` | Removed in LangChain 1.x. Use `langchain_core.tools.create_retriever_tool` and an inline prompt. |
| `model_decommissioned` error from Groq | The model name is retired. Use `openai/gpt-oss-20b` or another current Groq model. |
| Graph image does not render | `draw_mermaid_png()` needs internet access to mermaid.ink. Use `print(graph.get_graph().draw_mermaid())` for the text version. |
