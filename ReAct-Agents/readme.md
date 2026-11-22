# ReAct Agents with LangGraph: Groq for Reasoning, Hugging Face for Retrieval

This folder moves from **RAG chains** to **agents**. In the earlier folders (see [Hybrid-Retriever](../Hybrid-Retriever/readme.md)) every question ran the same fixed pipeline: retrieve, then answer. A **ReAct agent** lets the LLM decide for itself which steps to take: search Wikipedia, look in your private knowledge base, call a calculator, or answer straight away. It can also chain several steps, using one step's result in the next.

The notebook builds this up in stages:

1. **Tools**: Arxiv, Wikipedia, Tavily web search, three math functions, and a **RAG retriever tool** built on Hugging Face embeddings and FAISS.
2. **Tool calling**: a Groq LLM with `bind_tools` that *requests* tool calls but never runs them itself.
3. **The ReAct loop**: a LangGraph graph that runs the requested tools and feeds the results back to the LLM until it is done.
4. **Memory**: `MemorySaver` and `thread_id` so the agent remembers earlier turns.
5. **Swapping the LLM**: the same graph with a Hugging Face hosted model (`ChatHuggingFace`).
6. **The shortcut**: `create_agent`, LangChain's prebuilt version of the same loop.

```
ReAct-Agents/
├── code.ipynb   # Notebook: tools, ReAct graph, memory, HF model swap, prebuilt agent
└── readme.md    # This file
```

---

## 1. Core Concepts

### What "ReAct" means
**ReAct = Reason + Act** (Yao et al., 2022). The model works in a loop instead of answering in one shot:

| Step | What happens | In LangGraph |
|---|---|---|
| **Reason** | The LLM reads the whole conversation so far and decides the next move | `tool_calling_llm` node |
| **Act** | The LLM asks for a tool to be called with specific arguments | `AIMessage.tool_calls` |
| **Observe** | The tool runs and its output goes back into the conversation | `ToolNode` returns a `ToolMessage` |
| **Repeat / finish** | If more information is needed, loop; otherwise give the final answer | `tools_condition` routes to `tools` or `END` |

```
        ┌───────────────────────────────────────────┐
        ▼                                           │ observe (ToolMessage)
START ──► tool_calling_llm ──has tool_calls?──yes──► tools
                 │
                 no
                 ▼
                END
```

The edge from `tools` back to `tool_calling_llm` is what turns plain **tool calling** into an **agent**. Without it, the graph would run one round of tools and stop. With it, the model can see that `add(5, 5)` returned `10` and *then* decide to call `multiply(10, 10)`.

### Tools
A tool is a Python function plus a **name**, a **description** and an **argument schema**. The LLM only ever sees those three things, so:

- The **docstring is a prompt**. `acme_knowledge_base` says "Use this for any question about Acme Corp", and that sentence is what makes the model pick it.
- **Type hints become the schema**. `a: float, b: float` tells the model what JSON to send.

The `@tool` decorator builds all of this from the function automatically.

### The LLM requests, your code executes
`llm.bind_tools(tools)` sends the tool schemas with every request. When the model wants a tool it returns an `AIMessage` with empty `content` and a `tool_calls` list such as `[{'name': 'multiply', 'args': {'a': 12, 'b': 7}}]`. Nothing has run yet. `ToolNode` is the part that actually calls `multiply(12, 7)`.

### RAG as a tool (agentic RAG)
Wrapping the retriever in a tool changes **who decides when to retrieve**:

| Plain RAG chain | Agentic RAG |
|---|---|
| Retrieves on every question, even "Say hello in French" | Retrieves only when the model decides it is needed |
| One retrieval per question | Can retrieve several times with different queries |
| One knowledge source | Picks between your docs, Wikipedia, the web and so on |

### State and the `add_messages` reducer
The graph state is a single key, `messages`. Because it is annotated with `add_messages`, anything a node returns is **appended** to the list instead of replacing it. That growing list holds the question, every tool request and every tool result. It is the agent's scratchpad.

### Memory: checkpointers and threads
Without a checkpointer every `graph.invoke` starts from an empty state, so *"Divide that by 5"* fails because the agent has no idea what "that" is. Compiling with `checkpointer=MemorySaver()` saves the state after each step under a `thread_id`:

- Same `thread_id` → the new message is appended to the saved history.
- New `thread_id` → a fresh conversation.

`MemorySaver` keeps everything in RAM. For real apps use `SqliteSaver` or `PostgresSaver` so memory survives a restart.

---

## 2. Use Cases

- **Research assistants** that combine Arxiv papers, Wikipedia and live web search in one answer.
- **Customer support bots** that look up policy documents (RAG tool) and do calculations (refunds, totals) accurately.
- **Multi-step questions**, where the answer to step 1 is the input to step 2.
- **Any RAG system with several knowledge sources**, where the model should route each question to the right one.

---

## 3. Libraries Used

| Library | Used for |
|---|---|
| `langgraph` | `StateGraph`, `ToolNode`, `tools_condition`, `MemorySaver`: the agent loop and memory |
| `langchain-groq` | `ChatGroq`, the reasoning LLM (`openai/gpt-oss-20b`) |
| `langchain-huggingface` | `HuggingFaceEmbeddings` (local `all-MiniLM-L6-v2`) and `ChatHuggingFace` (hosted LLM) |
| `faiss-cpu` | Vector store behind the knowledge-base tool |
| `langchain-community` | `ArxivQueryRun`, `WikipediaQueryRun` |
| `langchain-tavily` | `TavilySearch` web search tool |
| `arxiv`, `wikipedia` | The clients those two tools wrap |
| `langchain` | `create_agent`, the prebuilt agent |

---

## 4. The Code, Cell by Cell

### Section 1: Install extra packages
Installs `arxiv<2.2` and `langchain-tavily`. The pin matters: `arxiv` 3.x removed `Search.results()`, which LangChain's Arxiv wrapper still calls, so newer versions crash with `AttributeError: 'Search' object has no attribute 'results'`.

### Section 2: Load API keys
Reads `.env` and maps `groq_api` / `tavily_api` to `GROQ_API_KEY` / `TAVILY_API_KEY`, the names the libraries expect. Turns on LangSmith tracing if `LANGCHAIN_API_KEY` is present, which lets you watch each Reason/Act step in the LangSmith UI.

### Section 3: Arxiv and Wikipedia tools
```python
wikipedia.set_user_agent("RAG-ReAct-Notebook/1.0 (learning project)")
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500))
```
`doc_content_chars_max` keeps tool output short, which saves tokens because every observation is sent back to the LLM. The custom user agent is needed because Wikipedia rate-limits (HTTP 429) the library's shared default one, and the wrapper then fails with a confusing `JSONDecodeError`.

### Section 4: Custom math tools
```python
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers a and b."""
    return a * b
```
`multiply.args` shows the generated schema: `{'a': {'type': 'number'}, 'b': {'type': 'number'}}`.

### Section 5: RAG retriever tool (Hugging Face + FAISS)
Four small "Acme Corp" documents are embedded with `all-MiniLM-L6-v2` (runs locally, no key) into FAISS. The retriever is then wrapped in a `@tool` whose docstring tells the model when to use it. Calling it directly with *"How long do I have to return a product?"* returns the refund-policy document.

### Section 6: Collect the tools
Builds the `tools` list and adds Tavily only if a key is set, so the notebook still runs without one.

### Section 7: Groq LLM and `bind_tools`
```python
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
llm_with_tools = llm.bind_tools(tools)
```
Two probes show both behaviours:
- *"What is 12 multiplied by 7?"* → `tool_calls = [{'name': 'multiply', 'args': {'a': 12, 'b': 7}}]`
- *"Say hello in French"* → `tool_calls = []`, `content = "Bonjour!"`

`temperature=0` makes tool choice more consistent. The original notebook's `qwen-qwq-32b` has been retired on Groq.

### Section 8: State schema
```python
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

### Section 9: Build the graph
```python
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)  # tools or END
builder.add_edge("tools", "tool_calling_llm")                        # the ReAct loop
```
`ToolNode(tools)` runs every tool call in the last `AIMessage`, in parallel if there are several, and returns one `ToolMessage` per call.

### Section 10: Run the agent
| Question | What the agent did |
|---|---|
| *Add 5 plus 5, then multiply the result by 10* | `add(5,5)` → 10, **then** `multiply(10,10)` → 100. Two loop iterations, because the second call needs the first result |
| *Acme's refund policy, and SmartPlug + SmartBulb cost?* | `acme_knowledge_base` → then `add(1499, 799)` → "30 days, ₹2,298" |
| *What is machine learning?* | Answered with or without `wikipedia`; the model decides |

### Section 11: The memory problem
*"What is 5 plus 8?"* → 13, then *"Divide that by 5"* → *"I'm not sure what number you'd like me to divide"*. Each call has its own empty state.

### Section 12: Memory with `MemorySaver`
On `thread_id="1"`: *Add 12 and 13* → 25, *Add that number to 25* → 50, *Then multiply that number by 2* → `multiply(50, 2)` → 100. The same question on `thread_id="2"` gets no context, which shows that threads are isolated.

### Section 13: Hugging Face LLM (optional)
```python
hf_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", task="text-generation"))
```
The same graph with a different model inside the node. It needs `HF_TOKEN` in `.env` and is skipped otherwise. Small open models choose tools less reliably than larger ones, so expect more mistakes on multi-step questions.

### Section 14: `create_agent`
```python
agent = create_agent(model=llm, tools=tools, system_prompt="...", checkpointer=MemorySaver())
```
One call builds the graph from sections 8, 9 and 12. Asked who founded Acme and how long ago, it retrieved "founded in 2015", worked out 11 years, and said honestly that the documents name no founder.

---

## 5. Limitations and Tips

- **Tool descriptions drive everything.** If the agent picks the wrong tool, rewrite the docstring before you touch anything else.
- **Every observation costs tokens.** Tool output is sent back to the LLM on every loop iteration, so keep `doc_content_chars_max` and `k` small.
- **Loops can run away.** The default step limit in this LangGraph version is very high (10,007), so set your own cap with `config={"recursion_limit": 10}`. Going past it raises `GraphRecursionError`.
- **The model can still do maths in its head.** A system prompt like "never do arithmetic in your head" (section 14) helps.
- **Memory grows forever.** Long threads eventually exceed the context window. Trim or summarise old messages in real apps.

---

## 6. Setup

```bash
pip install -r requirements.txt   # now includes arxiv<2.2 and langchain-tavily
```

`.env` in the repo root:
```
groq_api=gsk_...          # required, from console.groq.com
tavily_api=tvly-...       # optional, web search, from tavily.com
HF_TOKEN=hf_...           # optional, section 13, from huggingface.co/settings/tokens
LANGCHAIN_API_KEY=...     # optional, LangSmith tracing
```

---

## 7. Quick Glossary

| Term | Meaning |
|---|---|
| **Agent** | An LLM that chooses its own next steps (tools) in a loop |
| **ReAct** | Reason → Act → Observe, repeated until done |
| **Tool call** | A structured request from the LLM: tool name + JSON arguments |
| **`ToolMessage`** | The tool's result, added back into the conversation |
| **`ToolNode`** | LangGraph node that executes the requested tool calls |
| **`tools_condition`** | Router: go to `tools` if there are tool calls, else `END` |
| **Reducer (`add_messages`)** | Rule for merging node output into state (append, don't overwrite) |
| **Checkpointer** | Saves graph state after each step; gives the agent memory |
| **`thread_id`** | Key that separates one conversation's memory from another's |
| **Agentic RAG** | Retrieval exposed as a tool the agent calls only when needed |
