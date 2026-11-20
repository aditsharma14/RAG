# Agentic RAG with LangGraph: Groq for Generation, Hugging Face for Embeddings

This folder builds a **Retrieval-Augmented Generation (RAG) pipeline as a LangGraph graph**. Two blog posts are loaded from the web, split into chunks, embedded with a local Hugging Face model and stored in FAISS. A question then goes through two graph nodes: one **retrieves** the relevant chunks and one **generates** an answer from them with a Groq-hosted LLM.

The notebook is adapted from an OpenAI-based example. Every OpenAI component is replaced:

| Original (OpenAI) | This notebook |
|---|---|
| `init_chat_model("openai:gpt-4o")` | `init_chat_model("groq:openai/gpt-oss-20b")` |
| `OpenAIEmbeddings()` | `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")` |
| `OPENAI_API_KEY` | `GROQ_API_KEY` (read from `groq_api` in `.env`) and `HF_TOKEN` (read from `hf_token`) |
| `from langchain.schema import Document` | `from langchain_core.documents import Document` (the old path was removed in LangChain 1.x) |

```
Agentic-rag/
├── code.ipynb   # Notebook: web loading, chunking, FAISS index, LangGraph retrieve → answer graph
└── readme.md    # This file
```

---

## 1. Core Concepts

### What "Agentic RAG" means
**Agentic RAG** (Agentic Retrieval-Augmented Generation) is RAG where an LLM-driven **agent** controls the process instead of a fixed one-shot pipeline. An agent can:

- **reason** about the question,
- **plan** which steps to take,
- **retrieve** documents only when it needs them,
- **use tools** (search, calculators, retrievers),
- **retry or reflect** when a result is not good enough.

The result is an answer that is better grounded in the source documents.

### What this notebook actually builds
This notebook builds the **foundation** of an agentic system, not a full agent yet. The graph is a fixed, linear pipeline:

```
START ──► retriever ──► responder ──► END
```

Every question takes the same path. The LLM makes no decisions about *whether* to retrieve, and nothing loops back or retries. What the notebook teaches is the LangGraph structure that agentic behaviour is built on:

| LangGraph piece | Role here | What it enables later |
|---|---|---|
| **State** (`RAGState`) | Carries the question, retrieved documents and answer between steps | Extra fields such as a relevance grade or retry count |
| **Nodes** (`retrieve_docs`, `generate_answer`) | Plain Python functions that each do one step | New nodes such as "grade documents" or "rewrite query" |
| **Edges** (`add_edge`) | Fixed order: retriever, then responder | **Conditional edges** that let the LLM choose the next node |

Making this pipeline agentic means swapping a fixed edge for a conditional one, for example: *if the retrieved documents are not relevant, rewrite the question and retrieve again*. The [ReAct-Agents](../ReAct-Agents/readme.md) folder shows the other route, where the LLM calls the retriever as a tool inside a reasoning loop.

### What ReAct is
**ReAct (Reasoning + Acting)** is a framework where an LLM:

- **Reasons** step by step: breaks the question down and decides what to do next.
- **Acts** by calling tools such as search, calculators or retrievers.

This fits agentic RAG well, because the model can work through the cycle:

```
Think → Retrieve → Observe → Reflect → Final Answer
```

---

## 2. Setup

### API keys
Create a `.env` file in the repository root (it is loaded by `load_dotenv()`):

```env
groq_api=your_groq_api_key
hf_token=your_hugging_face_token
```

| Key | Used for | Where to get it |
|---|---|---|
| `groq_api` | The chat LLM (`openai/gpt-oss-20b` on Groq) | [console.groq.com/keys](https://console.groq.com/keys) |
| `hf_token` | Downloading the embedding model from the Hugging Face Hub. The model is public, so this mainly avoids rate limits and download warnings. | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

The notebook copies these values into the environment variable names the libraries expect (`GROQ_API_KEY`, `HF_TOKEN`).

### Packages
Installing the repository's `requirements.txt` covers these. `beautifulsoup4`, `langchain-text-splitters` and `pydantic` are not listed there by name but are pulled in as dependencies of the other packages. If an import fails, install them with `pip install beautifulsoup4 langchain-text-splitters`.

| Package | Why |
|---|---|
| `langchain`, `langchain-core` | `init_chat_model`, `Document` |
| `langchain-groq` | Backend that `init_chat_model("groq:...")` uses |
| `langchain-huggingface`, `sentence-transformers` | `HuggingFaceEmbeddings` running locally |
| `langchain-community`, `faiss-cpu` | `FAISS` vector store and `WebBaseLoader` |
| `beautifulsoup4` | HTML parsing inside `WebBaseLoader` |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| `langgraph` | `StateGraph`, `END` |
| `pydantic` | `BaseModel` for the graph state |
| `python-dotenv` | Loading `.env` |

---

## 3. Notebook Walkthrough

### Cell 1: Imports

```python
import os
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
```

| Import | Purpose |
|---|---|
| `List`, `BaseModel` | Type the fields of the graph state |
| `load_dotenv` | Read API keys from `.env` |
| `FAISS` | In-memory vector index for similarity search |
| `WebBaseLoader` | Download a web page and extract its text |
| `RecursiveCharacterTextSplitter` | Split long text into overlapping chunks |
| `init_chat_model` | Create a chat model from a `"provider:model"` string |
| `HuggingFaceEmbeddings` | Turn text into vectors with a local sentence-transformers model |
| `Document` | LangChain's container for `page_content` plus `metadata` |
| `StateGraph`, `END` | Build the LangGraph graph and mark where it finishes |

Running this cell prints two warnings, and neither stops the notebook:
- `langchain-community is being sunset`: the package still works, but its integrations are moving to standalone packages (for example `langchain-text-splitters` already has).
- `USER_AGENT environment variable not set`: see Cell 2.

### Cell 2: API keys and the LLM

```python
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("groq_api") or os.getenv("groq_api", "")
os.environ["HF_TOKEN"] = os.getenv("hf_token", "")
os.environ["USER_AGENT"] = "agentic-rag-notebook"

llm = init_chat_model("groq:openai/gpt-oss-20b")
llm
```

- `load_dotenv()` loads `.env` into the process environment.
- The lowercase keys from `.env` are copied to the uppercase names that `langchain-groq` and `huggingface_hub` look for. The `""` fallback stops `os.environ[...] = None` from raising a `TypeError` when a key is missing.
- `USER_AGENT` identifies the notebook's HTTP requests to the websites it downloads.
- `init_chat_model("groq:openai/gpt-oss-20b")` reads the `groq:` prefix, picks the `ChatGroq` class and connects to the open-weights **GPT-OSS 20B** model hosted on Groq. Displaying `llm` shows the resulting `ChatGroq` object and its model profile (131k token context window, tool-calling support, and so on).

### Cell 3: Load the source documents

```python
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = []
for url in urls:
    docs.extend(WebBaseLoader(url).load())

len(docs)
```

The two sources are Lilian Weng's blog posts **"LLM Powered Autonomous Agents"** and **"Diffusion Models for Video Generation"**. For each URL, `WebBaseLoader` downloads the HTML, extracts its text with BeautifulSoup and returns one `Document`. Its `metadata` holds the `source` URL, page `title`, `description` and `language`.

**Output:** `2`, one `Document` per page. The cell shows only the count because printing `docs` would print both full articles.

### Cell 4: Chunking, embedding and the FAISS index

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embedding)
retriever = vectorstore.as_retriever()

retriever.invoke("What are agents")
```

**1. Chunking.** `RecursiveCharacterTextSplitter` splits each article into pieces of at most **500 characters**. It tries to split on paragraph breaks first, then line breaks, then spaces, so sentences stay whole where possible. The **50-character overlap** repeats the end of each chunk at the start of the next one, so an idea that crosses a boundary is not lost. Each chunk keeps its parent document's metadata.

Small chunks matter for retrieval: a question should match a focused passage, not a whole article.

**2. Embedding.** `all-MiniLM-L6-v2` is a small sentence-transformers model (about 90 MB, 384-dimensional vectors) that runs on your own CPU. It is downloaded from the Hugging Face Hub the first time (the `Loading weights` progress bar) and cached after that. Text with similar meaning gets vectors that are close together.

**3. Indexing.** `FAISS.from_documents` embeds every chunk and stores the vectors in an in-memory FAISS index, keeping the original `Document` alongside each one.

**4. Retriever.** `as_retriever()` wraps the vector store in LangChain's retriever interface. A query is embedded with the same model and the **4 nearest chunks** (the default `k`) are returned.

**Output:** the test query `"What are agents"` returns four chunks from the *LLM Powered Autonomous Agents* post. This confirms the index works before it is wired into the graph.

### Cell 5: The graph state

```python
class RAGState(BaseModel):
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
```

The **state** is the shared data that flows through the graph. Each node reads from it and returns updates to it.

| Field | Set by | Contents |
|---|---|---|
| `question` | The caller, when invoking the graph | The user's question |
| `retrieved_docs` | `retriever` node | The chunks returned by FAISS |
| `answer` | `responder` node | The LLM's final answer |

Because `RAGState` is a Pydantic model, LangGraph validates the input: invoking the graph without a `question` raises a validation error. The defaults (`[]` and `""`) let the graph start with only the question filled in. Pydantic copies mutable defaults for each instance, so `retrieved_docs = []` is not shared between runs.

### Cell 6: The nodes

```python
def retrieve_docs(state: RAGState):
    docs = retriever.invoke(state.question)
    return {"retrieved_docs": docs}

def generate_answer(state: RAGState):
    context = "\n\n".join(doc.page_content for doc in state.retrieved_docs)
    prompt = (
        "Answer the question based on the context.\n\n"
        f"Context:\n{context}\n\nQuestion: {state.question}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}
```

A node is a function that takes the current state and returns a **partial update**: a dict of only the fields it changed. LangGraph merges the update into the state, and every other field keeps its value.

- **`retrieve_docs`** sends the question to the FAISS retriever and stores the top 4 chunks in `retrieved_docs`.
- **`generate_answer`** joins the text of those chunks with blank lines between them, builds a prompt that places the context before the question, sends it to the Groq LLM and stores `response.content` (the text of the returned `AIMessage`) in `answer`.

The original example returned a whole new `RAGState(...)` from each node, which meant repeating every field and risked resetting one by accident. Returning only the changed fields avoids that.

The prompt tells the model to answer "based on the context". This is what makes the answer **grounded**: the model is pointed at the retrieved passages instead of relying only on what it learned in training.

### Cell 7: Build and compile the graph

```python
builder = StateGraph(RAGState)

builder.add_node("retriever", retrieve_docs)
builder.add_node("responder", generate_answer)

builder.set_entry_point("retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)

graph = builder.compile()
graph
```

| Call | What it does |
|---|---|
| `StateGraph(RAGState)` | Creates a graph whose state has the shape of `RAGState` |
| `add_node(name, fn)` | Registers a function as a named step |
| `set_entry_point("retriever")` | Makes `retriever` the first node to run (same as `add_edge(START, "retriever")`) |
| `add_edge("retriever", "responder")` | After `retriever` finishes, run `responder` |
| `add_edge("responder", END)` | After `responder`, the graph stops |
| `compile()` | Checks the structure (no unreachable nodes, a valid entry point) and returns a runnable `CompiledStateGraph` |

In Jupyter, displaying `graph` normally renders a diagram of the graph. If the image cannot be rendered, the text form `<langgraph.graph.state.CompiledStateGraph object ...>` is shown instead. To draw it explicitly:

```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

### Cell 8: Run the graph

```python
user_question = "What is the concept of agent loop in autonomous agents?"
final_state = graph.invoke(RAGState(question=user_question))

print("✅ Final Answer:\n", final_state["answer"])
```

`graph.invoke` runs the nodes in order and returns the final state. The returned state is a **plain `dict`**, not a `RAGState` object, so the answer is read with `final_state["answer"]`, not `final_state.answer`. `final_state["retrieved_docs"]` holds the chunks the answer was based on, which is useful for checking where the answer came from.

**Output:** the model describes the agent loop as a repeating cycle of steps: perceive/observe, update internal state and memory, plan/decide, act, observe the outcome, and learn/update. It presents them as a table and draws on the retrieved passages from the agents blog post.

---

## 4. End-to-End Data Flow

```
                 ┌──────────── indexing (runs once) ────────────┐
 2 blog URLs ──► WebBaseLoader ──► 2 Documents ──► splitter (500/50)
                                                     │
                                                     ▼
                                   chunks ──► all-MiniLM-L6-v2 ──► FAISS index
                 └───────────────────────────────────────────────┘
                                                                     │
 ┌──────────────── query time (LangGraph) ────────────────┐          │
 │ {question}                                             │          │
 │    │                                                   │          │
 │    ▼                                                   │          │
 │ retriever node ── retriever.invoke(question) ──────────┼──────────┘
 │    │  {retrieved_docs: top 4 chunks}                   │
 │    ▼                                                   │
 │ responder node ── prompt(context + question) ──► Groq gpt-oss-20b
 │    │  {answer}                                         │
 │    ▼                                                   │
 │   END ──► final_state["answer"]                        │
 └────────────────────────────────────────────────────────┘
```

---

## 5. Limitations and Next Steps

| Limitation | Effect | Possible improvement |
|---|---|---|
| Always retrieves | Even small talk triggers a vector search | Add a routing node that decides whether retrieval is needed |
| No relevance check | Irrelevant chunks still go into the prompt | Add a "grade documents" node with a conditional edge |
| No retry | A poor first retrieval gives a poor answer | Rewrite the query and loop back to `retriever` when grading fails |
| No memory | Each `invoke` is independent | Compile with a `MemorySaver` checkpointer and pass a `thread_id` |
| Index rebuilt on every run | Pages are re-downloaded and re-embedded each time | Save with `vectorstore.save_local(...)` and reload with `FAISS.load_local(...)` |
| No source citations | The user cannot see where the answer came from | Print `doc.metadata["source"]` for each entry in `final_state["retrieved_docs"]` |

The first three rows are what would turn this pipeline into a true agentic RAG system.

---

## 6. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `USER_AGENT environment variable not set` warning | `WebBaseLoader` reads `USER_AGENT` when `langchain_community.document_loaders` is **imported**. Cell 2 sets it after the import in Cell 1, so the warning still shows. Move the `os.environ["USER_AGENT"] = ...` line above the imports to silence it. The warning is harmless. |
| `ImportError: cannot import name 'Document' from 'langchain.schema'` | You are using the old import path. Use `from langchain_core.documents import Document`. |
| Groq authentication error | `groq_api` is missing from `.env` or the notebook was started from a folder where `load_dotenv()` cannot find the file. |
| `AttributeError: 'dict' object has no attribute 'answer'` | `graph.invoke` returns a dict. Use `final_state["answer"]`. |
| Slow first run of Cell 4 | The embedding model is being downloaded. Later runs load it from the local cache. |
