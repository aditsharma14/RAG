# Multi-Agent RAG with LangGraph: Groq + Hugging Face + Tavily

This folder builds a **multi-agent RAG system**: instead of one LLM doing everything, the work is split between specialised agents, each with its own prompt and tools, which collaborate on one request through a shared message history.

The notebook's main example is a **network** of two agents that write a blog together:

| Agent | Tools | Job |
|---|---|---|
| `researcher` | `InternalResearchNotes` (FAISS over `internal_docs.txt`) and `web_search` (Tavily) | Collects short research notes |
| `blog_generator` | None | Turns the notes into a ~1,000-word blog and ends with `FINAL ANSWER` |

```
MultiAgent/
├── code.ipynb           # Notebook: tools, network of researcher + blog writer, start of the supervisor example
├── internal_docs.txt    # Fictional engineering report on transformer models in production (NovaMind Labs)
└── readme.md            # This file
```

---

## 1. Core Concepts

### What is a multi-agent RAG system?
A normal RAG pipeline is one chain: retrieve, then generate. A multi-agent system gives separate roles to separate agents:

- Each agent is a **ReAct agent** (`create_agent`): the LLM calls a tool, reads the result, and repeats until it can answer.
- A **retriever can be a tool**. The vector store is wrapped in a `Tool`, so the agent decides when to search internal documents and when to search the web.
- Agents communicate through a shared `messages` list (`MessagesState`).

### Three common architectures

| Pattern | Who decides the next step? | Shape |
|---|---|---|
| **Network** (this notebook) | Each agent. An agent that has the final result writes `FINAL ANSWER`; otherwise it hands over to the other agent | `researcher ⇄ blog_generator` |
| **Supervisor** | A central supervisor LLM picks which worker runs next. Workers always report back to it | `supervisor → {research_agent, math_agent} → supervisor` |
| **Hierarchical teams** | A top-level supervisor routes between teams. Each team is its own subgraph with a mid-level supervisor | `top supervisor → {research team, writing team}` |

The network is the simplest but least controlled: if no agent ever says `FINAL ANSWER`, the agents keep handing the task back and forth. Supervisors give central control, and hierarchies help when one supervisor would have too many workers.

### `Command`: update and route in one return
Each node returns a `Command` instead of relying on fixed edges:

```python
return Command(
    update={"messages": [HumanMessage(content=last.content, name="researcher")]},
    goto="blog_generator",   # or END
)
```

`update` changes the shared state, and `goto` names the next node. The only fixed edge in the graph is `START → researcher`.

The agent's final message is stored as a `HumanMessage` with a `name`, because some providers reject an AI message in the last position of the input.

### Staying inside Groq's free-tier token limit
On the free tier, `openai/gpt-oss-20b` allows **8,000 tokens per minute**, and a single request larger than that fails with `413 Request too large`. Groq also counts `max_tokens` as part of each request. Multi-agent graphs hit this quickly, because every tool result and every agent reply is added to the history.

The notebook keeps requests small in five ways:

| Technique | Where | What it prevents |
|---|---|---|
| `compact(state)`: give each agent only the user's question and the latest message | Cells 3–5 | The history growing with every round |
| Store only the agent's **final** message in the shared state, not its tool calls and tool results | `research_node`, `blog_node` | Raw search results reaching the next agent |
| `ToolCallLimitMiddleware(run_limit=3)` | `tool_limit()` in Cell 2 | One agent run collecting many tool results |
| `web_search`: 3 Tavily results, 600 characters each; retriever `k=3` | Cell 2 | Large tool outputs |
| `max_tokens=1500` for `llm`, `max_tokens=3500` + `reasoning_effort="low"` for `writer_llm`, `max_retries=5` | Cell 1 | Oversized requests; `429` errors are retried automatically |

---

## 2. Setup

### API keys
`.env` in the repository root:

```env
groq_api=your_groq_api_key
hf_token=your_hugging_face_token
tavily_api=your_tavily_api_key
```

The notebook copies these into `GROQ_API_KEY`, `HF_TOKEN` and `TAVILY_API_KEY`. Do not write keys directly in the notebook.

### Packages
From the repository's `requirements.txt`: `langchain`, `langchain-groq`, `langchain-huggingface`, `sentence-transformers`, `langchain-community`, `langchain-tavily`, `faiss-cpu`, `langgraph`, `python-dotenv`.

The supervisor example also needs:

```
pip install langgraph-supervisor
```

The notebook uses the **langchain 1.x** API: `langchain.agents.create_agent(..., system_prompt=...)` replaces `langgraph.prebuilt.create_react_agent(..., prompt=...)`.

---

## 3. Knowledge Base: `internal_docs.txt`

A fictional internal engineering report from "NovaMind Labs" (about 440 lines). Unlike the shorter report in [AnswerSynthesis](../AnswerSynthesis/readme.md), it includes many concrete numbers, so research and math questions have exact answers to find.

| Section | Contents |
|---|---|
| 1. Executive summary | Model counts, monthly requests (~80 million), cost (61,400 USD), p95 latency, uptime, top wins and risks |
| 2. Production models | EfficientFormer, DistilBERT, TinyBERT, LLaMA 2 13B, Mistral 7B, Whisper Small, MiniLM cross-encoder: use case, deployment, metrics, cost, issues |
| 3. Rollout and evaluation | Swin Transformer, LLaMA 3 8B, Reformer, ViT, Longformer, Phi-3 Mini |
| 4. RAG pipeline "Atlas" | Ingestion, embeddings, FAISS, hybrid retrieval, reranking, multi-agent orchestration |
| 5. Infrastructure | Triton, vLLM, EKS, monitoring, canary rollout, CI/CD |
| 6. Optimizations | Distillation, LoRA, FlashAttention-2, RoPE scaling, AWQ, tokenizers, roadmap |
| 7. Cost table | Monthly requests and cost per model; totals match the summary |
| 8. Incidents | Five incidents, including a multi-agent loop, with fixes and lessons |
| 9. Glossary | Short definitions |

Example questions it can answer: *"Which transformer variants are in production?"*, *"How much did the move to Mistral 7B save?"*, *"What is the combined monthly cost of LLaMA 2 13B and Mistral 7B?"*

---

## 4. Notebook Walkthrough

### Cell 1: Environment, LLMs and embeddings
Loads `.env`, sets the three keys and `USER_AGENT`, and asserts the Groq and Tavily keys.

```python
llm        = ChatGroq(model="openai/gpt-oss-20b", temperature=0,   max_tokens=1500, max_retries=5)
writer_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3, max_tokens=3500,
                      reasoning_effort="low", max_retries=5)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

`writer_llm` exists because a 1,000-word blog does not fit in 1,500 output tokens. `reasoning_effort="low"` matters: `gpt-oss-20b` also spends output tokens on hidden reasoning, and in tests with a large `max_tokens` and default reasoning, the blog sometimes came back empty or cut off.

### Cell 2: Tools
- **`web_search`**: a `@tool` wrapper around `TavilySearch(max_results=3)` that returns only title, URL and the first 600 characters of each result. Its schema has one `query` field, which is much smaller than `TavilySearch`'s own schema. `tavily_tool` is kept as an alias.
- **`make_retriever_tool_from_text`**: loads a text file, splits it into 500/50 character chunks, builds a FAISS index with the Hugging Face embeddings, and wraps the retriever (`k=3`) in a `Tool`.
- **`tool_limit(n)`**: returns `[ToolCallLimitMiddleware(run_limit=n)]` for `create_agent(..., middleware=...)`.
- **`internal_tool_1`**: the `InternalResearchNotes` tool over `internal_docs.txt`. The cell prints the first 500 characters of a test search.

### Cell 3: Helpers
- **`get_next_node(last_message, goto)`**: returns `END` if the message contains `FINAL ANSWER`, otherwise `goto`.
- **`make_system_prompt(suffix)`**: the shared collaboration prompt: use your tools, hand over what you cannot finish, prefix the final deliverable with `FINAL ANSWER`.
- **`compact(state)`**: returns only the first message (the user's request) and the last message (the other agent's latest output).

### Cell 4: Research agent and node
`create_agent(llm, tools=[internal_tool_1, tavily_tool], middleware=tool_limit(3), ...)`. The prompt asks for at most 3 tool calls and bullet-point notes under 300 words, and tells the researcher **not** to write the blog or say `FINAL ANSWER`. Without that line, the researcher sometimes wrote the blog itself and ended the graph before the writer ran.

### Cell 5: Blog agent and node
`create_agent(writer_llm, tools=[], ...)`, asked for a ~1,000-word blog that starts with `FINAL ANSWER`.

`blog_node` also stops the graph when the reply is longer than 3,000 characters, even without `FINAL ANSWER`. The model does not always follow that instruction: in one test it wrote three full blogs, because the first two did not contain the marker and were sent back to the researcher.

### Cell 6: Graph

```python
workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("blog_generator", blog_node)
workflow.add_edge(START, "researcher")
graph = workflow.compile()
```

### Cell 7: Run

```python
response = graph.invoke(
    {"messages": [("user", "Write a detailed blog on transformer variants in production deployments. "
                           "Keep research notes short.")]},
    config={"recursion_limit": 12},
)
```

`recursion_limit` is a safety net: if the agents keep handing the task back and forth, LangGraph stops with an error instead of running forever.

**Test runs:**

| Run | Result |
|---|---|
| Before the length-based stop | 3 research rounds and 3 blogs, 210 s, no errors. The first two blogs had no `FINAL ANSWER` |
| One-round run | researcher → blog_generator → `END` in 28 s: notes of ~2,300 characters, blog of ~8,300 characters starting with `FINAL ANSWER` |

Typical tool use per research run: 2 internal searches and 1–2 web searches.

### Cell 8: Supervisor example: worker agents
Defines the workers for the supervisor pattern:

- **`research_agent`** (`name="research_agent"`): `web_search` + `internal_tool_1`, tool limit 3, research only.
- **`math_agent`** (`name="math_agent"`): `add`, `multiply`, `divide`, math only.

The `name` argument matters: `langgraph-supervisor` uses it to create a handoff tool for each worker (for example `transfer_to_math_agent`). The supervisor itself is the next step:

```python
from langgraph_supervisor import create_supervisor

supervisor = create_supervisor(
    model=llm,
    agents=[research_agent, math_agent],
    prompt="...assign research tasks to the research agent and math tasks to the math agent, one at a time...",
    add_handoff_back_messages=True,
    output_mode="last_message",   # keep only each worker's final answer, to save tokens
).compile()
```

---

## 5. End-to-End Data Flow (Network)

```
 user request
      │
      ▼
 researcher ── compact(state) ──► ReAct loop (≤ 3 tool calls)
      │                              ├── InternalResearchNotes → FAISS(internal_docs, k=3)
      │                              └── web_search → Tavily (3 × 600 chars)
      │        only the final notes are added to the shared state
      ▼
 blog_generator ── compact(state) ──► writer_llm (≤ 3,500 tokens)
      │
      ├── "FINAL ANSWER" or blog > 3,000 chars ──► END
      └── otherwise ──► researcher (another round)
```

---

## 6. Limitations and Next Steps

| Limitation | Effect | Possible improvement |
|---|---|---|
| `FINAL ANSWER` depends on the model following instructions | Extra rounds; the length check is a workaround | Use a supervisor or a structured "done" field |
| `compact` hides older messages | The writer sees only the latest notes, not earlier rounds | Summarise older rounds instead of dropping them |
| The blog is not checked against the sources | Web content and internal facts can be mixed without citations | Add a citation rule or a grounding check, as in [SelfReflection](../SelfReflection/readme.md) |
| Internal notes and web results are not labelled | The writer may present internal numbers as public facts | Tag tool output (`[INTERNAL]`, `[WEB]`) as in [AnswerSynthesis](../AnswerSynthesis/readme.md) |
| Free-tier token limit | Long runs wait on `429` retries | A model or tier with a higher tokens-per-minute limit |
| Supervisor and hierarchical examples | Workers defined; supervisor not built yet | Add `create_supervisor`, then team subgraphs with a top-level supervisor |

---

## 7. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `413 Request too large ... Limit 8000, Requested 9688` | One request is over the per-minute limit. Check that the nodes use `compact(state)`, store only the final message, and that `tool_limit` and the short `web_search` are used. Lower `[:600]` or `max_results` if needed. |
| `429 Rate limit reached ... Please try again in 23s` | Several requests in the same minute. `max_retries=5` waits and retries automatically; the run just takes longer. |
| Blog is empty or stops mid-sentence | `max_tokens` too small, or reasoning used the tokens. Use `writer_llm` with `max_tokens=3500` and `reasoning_effort="low"`. |
| The researcher writes the blog itself | Its prompt must say not to write the blog and not to say `FINAL ANSWER`. |
| Agents loop until `GraphRecursionError` | Nobody wrote `FINAL ANSWER`. Keep the length-based stop in `blog_node`, or raise `recursion_limit`. |
| `APIConnectionError: Connection error` | A temporary network problem reaching Groq. Rerun the cell. |
| `ModuleNotFoundError: langgraph_supervisor` | `pip install langgraph-supervisor` in the notebook's environment. |
| `ImportError` for `langchain.vectorstores`, `langchain.text_splitter` or `langchain.agents.Tool` | Old import paths removed in langchain 1.x. Use `langchain_community.vectorstores`, `langchain_text_splitters` and `langchain_core.tools.Tool`. |
