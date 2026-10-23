# MMR for RAG: Retrieving Chunks That Are Relevant *and* Different

This folder covers **Maximal Marginal Relevance (MMR)**, a retrieval strategy that picks chunks which are relevant to the question but also **different from each other**. Plain similarity search (see [LangChain-FAISS](../LangChain-FAISS/) and [Embeddings](../Embeddings/readme.md)) returns the *k* chunks closest to the question, and on real data those are often near-copies of one another. MMR trades a little relevance for variety, so the few chunks that fit in the prompt cover more ground.

The notebook builds a full RAG pipeline with an MMR retriever in five steps:

1. **Load and chunk a text file**: `TextLoader` plus `RecursiveCharacterTextSplitter` turn `langchain_rag_dataset.txt` into 10 chunks.
2. **Vector store**: FAISS with `all-MiniLM-L6-v2` embeddings.
3. **MMR retriever**: `as_retriever(search_type="mmr", search_kwargs={"k": 3})`.
4. **Answer chain**: `create_stuff_documents_chain` pastes the retrieved chunks into a prompt for a Groq-hosted LLM.
5. **Retrieval chain**: `create_retrieval_chain` joins the retriever and the answer chain, so one `invoke` goes from question to answer.

```
MMR/
├── code.ipynb                  # Notebook: FAISS + MMR retriever + RAG answer chain
├── langchain_rag_dataset.txt   # 20 sentences about LangChain, RAG, retrievers, chains, agents, memory
└── readme.md                   # This file
```

---

## 1. Core Concepts

### The problem: redundant results
Similarity search scores every chunk **on its own** against the question. It never looks at what it has already picked. If a knowledge base has five chunks that say almost the same thing, and they are the five closest to the question, similarity search returns all five, and the LLM gets one fact five times instead of five facts.

```
Question: "How does LangChain support agents and memory?"

Similarity top-3                 MMR top-3
1. memory + agents (overview)    1. memory + agents (overview)
2. agents + memory classes       2. agents use APIs and databases   ← new information
3. memory + agents (intro line)  3. agents + memory classes
   └─ mostly repeats #1
```

### Maximal Marginal Relevance
**Definition:** A greedy selection method that adds chunks one at a time. Each step picks the candidate with the best balance of "close to the question" and "far from the chunks already picked".

For each candidate chunk `d`, with `S` the chunks already selected:

```
MMR(d) = λ · sim(question, d)  −  (1 − λ) · max over s in S of sim(d, s)
         └──── relevance ────┘     └──────── redundancy ────────┘
```

The algorithm:

1. Fetch the `fetch_k` most similar chunks to the question (the candidate pool).
2. Pick the single most similar chunk first.
3. For every remaining candidate, compute `MMR(d)`. A candidate that is very similar to something already picked gets a large penalty.
4. Pick the highest-scoring candidate, add it to `S`, and repeat until `k` chunks are picked.

"Marginal" means each chunk is judged by what it **adds** on top of the chunks already chosen.

### The three knobs

| Parameter | Default in LangChain | What it controls |
|---|---|---|
| `k` | 4 | How many chunks are finally returned (3 in this notebook) |
| `fetch_k` | 20 | How many candidates are fetched by similarity before MMR re-selects from them |
| `lambda_mult` (λ) | 0.5 | The balance: `1.0` = pure relevance (same as similarity search), `0.0` = pure diversity |

### Similarity vs MMR

| | Similarity search | MMR |
|---|---|---|
| Scores each chunk against | The question only | The question **and** the chunks already picked |
| Result | The k closest chunks, often overlapping | k chunks that are relevant and cover different points |
| Extra cost | None | A few extra vector comparisons on `fetch_k` candidates (no extra model call) |
| Best when | Each chunk is unique, or one exact match is needed | The data has repeats, or the question has several parts |

---

## 2. Use Cases

MMR helps when the **variety** of the retrieved chunks matters:

- **Repetitive sources**: FAQs, product docs, release notes and scraped web pages often say the same thing in several places. MMR keeps one copy and uses the other slots for new information.
- **Overlapping chunks**: a large `chunk_overlap` makes neighbouring chunks share text. Similarity search tends to return several neighbours together; MMR spreads out.
- **Multi-part or broad questions**: *"How does LangChain support agents and memory?"* has two topics. MMR is more likely to bring back chunks for both instead of three chunks about one.
- **Summaries and overviews**: when the LLM should cover a topic from several angles, diverse context gives a fuller answer.
- **Small context budgets**: with only 3–4 slots in the prompt, a near-duplicate wastes a whole slot.

It is **less useful** when:
- The question has one exact answer in one chunk. Diversity can push a second, supporting chunk out in favour of a less relevant one.
- The data is small and clean with no repeats, so there is nothing redundant to remove.
- λ is set too low. The retriever starts returning chunks that are different but off-topic.

---

## 3. Libraries Used

| Library | Purpose in this notebook |
|---|---|
| `langchain-community` | `TextLoader` to read the `.txt` file and the `FAISS` vector store wrapper (which implements MMR search) |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` to cut the text into chunks |
| `langchain-huggingface` | `HuggingFaceEmbeddings`, the LangChain wrapper around sentence-transformers |
| `sentence-transformers` | Downloads and runs the `all-MiniLM-L6-v2` embedding model locally |
| `faiss-cpu` | The vector index that does the fast similarity search |
| `langchain-core` | `PromptTemplate`, `Document` |
| `langchain-classic` | `create_stuff_documents_chain` and `create_retrieval_chain` (moved here from `langchain` in LangChain 1.x) |
| `langchain` | `init_chat_model`, a single function that creates a chat model from a `"provider:model"` string |
| `langchain-groq` | `ChatGroq`, created behind the scenes by `init_chat_model("groq:...")` |
| `python-dotenv` | Loads the Groq API key from a `.env` file |
| `langchain-openai` | Imported (`ChatOpenAI`) but not used; the notebook uses Groq instead |

---

## 4. The Code, Cell by Cell

### Cell 0: Imports

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI   # or from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
```

| Import | What it is |
|---|---|
| `FAISS` | LangChain's wrapper around the FAISS index. It provides both `similarity_search` and `max_marginal_relevance_search`. |
| `HuggingFaceEmbeddings` | Turns text into vectors with a local sentence-transformers model. |
| `TextLoader` | Reads a plain text file into a list with one `Document`. |
| `RecursiveCharacterTextSplitter` | Splits text into chunks of a maximum size, preferring to break at paragraphs, then lines, then spaces. |
| `ChatOpenAI` | The OpenAI chat client. Imported but **not used**. |
| `PromptTemplate` | A reusable prompt with `{placeholders}` that get filled in at run time. |
| `Document` | LangChain's text container (`page_content` + `metadata`). Imported but not used directly. |
| `create_stuff_documents_chain` | Builds a chain that "stuffs" a list of documents into one prompt variable and calls the LLM. |
| `create_retrieval_chain` | Builds a chain that runs a retriever first, then passes its documents to a documents chain. |

The `DeprecationWarning` in the output says `langchain-community` is being sunset. The code still works; in new projects, prefer the standalone integration packages.

---

### Cell 1: Load the API key

```python
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("groq_api")
```

- `load_dotenv()` reads the `.env` file (in the current or a parent folder) and puts its `KEY=value` lines into the environment.
- The last line copies the key from the `.env` name (`groq_api`) to the name `ChatGroq` looks for (`GROQ_API_KEY`). If `groq_api` is missing, `os.getenv` returns `None` and this line raises a `TypeError`.

---

### Cell 2: Load and chunk the text file

```python
loader = TextLoader("langchain_rag_dataset.txt")
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(raw_docs)
chunks
```

| Line | What it does |
|---|---|
| `TextLoader("langchain_rag_dataset.txt")` | Points at the dataset. The path is relative, so run the notebook from inside `MMR/`. |
| `chunk_size=300` | Each chunk is at most 300 characters. |
| `chunk_overlap=50` | Up to 50 characters may repeat between chunks. Every line in this file is longer than 50 characters and the splitter breaks at line ends, so in practice no text is repeated here. |

The file has 20 sentences, one per line. The splitter packs 2 (sometimes 3) lines into each chunk, which gives **10 chunks**:

| # | Chunk content (short) |
|---|---|
| 1 | LangChain is an open-source framework… abstractions for prompts, chains, **memory, and agents** |
| 2 | Integration with FAISS and Chroma… LangChain enables RAG |
| 3 | **Memory** helps models retain interactions… **Agents** can use tools like calculators, search APIs |
| 4 | BM25 + vector retrieval for hybrid search… FAISS for similarity search |
| 5 | Chroma is a lightweight vector store… prompt templates with Jinja-style formatting |
| 6 | The 'stuff', 'map-reduce' and 'refine' chains |
| 7 | LLMs act as **agents** that decide which tool to call… ConversationBufferMemory / ConversationSummaryMemory (**memory**) |
| 8 | **Agents** interact with external APIs and databases… RAG pipeline steps |
| 9 | MMR improves diversity… tool usage lets agents execute Python functions |
| 10 | Reranking with LLMs or cross-encoders |

The numbers are just labels for this readme, so the chunks can be followed through the next cells.

---

### Cell 3: Vector store

```python
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
```

| Line | What it does |
|---|---|
| `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` | Loads the small, free embedding model (384-number vectors, runs on CPU). The first run downloads about 90 MB. |
| `FAISS.from_documents(chunks, embedding_model)` | Embeds all 10 chunks and builds an in-memory FAISS index, keeping the text and metadata next to each vector. |

The `Loading weights: 0/103` progress bar is normal.

---

### Cell 4: The MMR retriever

```python
retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)
```

This is the only line that differs from a normal RAG pipeline.

| Argument | What it does |
|---|---|
| `search_type="mmr"` | Makes the retriever call `max_marginal_relevance_search` instead of `similarity_search`. |
| `search_kwargs={"k":3}` | Return 3 chunks. `fetch_k` and `lambda_mult` are left at their defaults (20 and 0.5). |

With only 10 chunks and `fetch_k=20`, **every** chunk is in the candidate pool, and MMR chooses 3 of the 10. On a real dataset `fetch_k` would be much smaller than the total, e.g. `{"k": 4, "fetch_k": 25, "lambda_mult": 0.6}`.

---

### Cell 5: Prompt and LLM

```python
from langchain.chat_models import init_chat_model
prompt = PromptTemplate.from_template("""
Answer the question based on the context provided.

Context:
{context}

Question: {input}
""")
llm=init_chat_model("groq:openai/gpt-oss-20b")
```

| Part | Why it is there |
|---|---|
| `{context}` | Filled by `create_stuff_documents_chain` with the retrieved chunks. The name `context` is what that function expects by default. |
| `{input}` | Filled with the user's question. `create_retrieval_chain` expects the question under the key `input`. |
| `init_chat_model("groq:openai/gpt-oss-20b")` | A provider-neutral factory. The part before `:` picks the provider (`groq` → `ChatGroq`), the rest is the model name: OpenAI's open-weight 20B reasoning model hosted on Groq. |

---

### Cell 6: Build the chains

```python
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
```

```
{"input": "How does LangChain support agents and memory?"}
      ▼
retriever (MMR, k=3)          → 3 Documents, stored under "context"
      ▼
document_chain
   ├─ joins the 3 page_contents with blank lines → {context}
   ├─ prompt fills {context} and {input}
   └─ llm → answer text, stored under "answer"
      ▼
{"input": ..., "context": [Document, Document, Document], "answer": "..."}
```

"Stuff" means all retrieved chunks go into the prompt at once, which is fine for 3 short chunks.

---

### Cell 7: Ask the question

```python
query = {"input": "How does LangChain support agents and memory?"}
response = rag_chain.invoke(query)
print("✅ Answer:\n", response["answer"])
```

The answer is a Markdown table with rows for **Agents**, **Memory** and **RAG & Retrieval**, followed by a short summary. It names `ConversationBufferMemory` and `ConversationSummaryMemory` (from chunk 7) and agents calling calculators, search APIs and custom functions (from chunk 3).

---

### Cell 8: Inspect the full response

```python
response
```

The dict shows the 3 chunks MMR picked, in the order it picked them:

| MMR pick | Chunk |
|---|---|
| 1 | **3**: memory helps retain interactions + agents use tools |
| 2 | **8**: agents interact with external APIs and databases + RAG pipeline steps |
| 3 | **7**: agents decide which tool to call + memory classes |

---

## 5. Similarity vs MMR on This Data

To see what MMR changed, the same question was run through plain similarity search on the same index (`vectorstore.similarity_search_with_score(query, k=10)`). FAISS returns a distance, so **lower means closer**:

| Similarity rank | Chunk | Distance |
|---|---|---|
| 1 | 3: memory + agents use tools | 0.346 |
| 2 | 7: agents decide tools + memory classes | 0.567 |
| 3 | 1: framework intro, "prompts, chains, memory, and agents" | 0.709 |
| 4 | 8: agents use APIs and databases + RAG pipeline | 0.800 |
| 5 | 9: MMR + tool usage | 1.087 |

#### Top 3: similarity vs MMR

| Rank | Similarity (`k=3`) | MMR (`k=3`, λ = 0.5) |
|---|---|---|
| 1 | 3: memory + agents use tools | 3: memory + agents use tools |
| 2 | 7: agents decide tools + memory classes | **8: agents use APIs and databases** ← new |
| 3 | 1: framework intro (memory, agents) | 7: agents decide tools + memory classes |

- MMR **dropped chunk 1**. It is the third most similar, but it only *lists* memory and agents, which chunk 3 already covers, so its redundancy penalty was high.
- MMR **added chunk 8**, which brings new information (agents calling external APIs and databases).
- The order shows the greedy selection: after chunk 3, chunk 8 was the best mix of relevant and *different*, so it was picked before chunk 7.

#### Effect of λ (`max_marginal_relevance_search(query, k=3, lambda_mult=...)`)

| `lambda_mult` | Chunks returned | What happens |
|---|---|---|
| 1.0 | 3, 7, 1 | Pure relevance, identical to similarity search |
| 0.5 (default) | 3, 8, 7 | Relevant, with one more varied chunk |
| 0.25 | 3, 4, 5 | Too much diversity: hybrid retrieval and Chroma/prompt templates are off-topic |
| 0.0 | 3, 4, 5 | Pure diversity after the first pick |

λ is the most important setting: too low and MMR fills the prompt with unrelated chunks.

#### Side effect in the answer
The extra variety shows up in the answer: the **RAG & Retrieval** row comes from the second half of chunk 8 ("RAG pipelines in LangChain involve document loading, splitting…"), which has nothing to do with agents or memory. Diversity brings in new information, and some of it can be off-topic.

---

## 6. Limitations and Tips

- **Tune `lambda_mult`.** 0.5 is a reasonable start. Raise it (0.7–0.8) for precise questions, lower it for broad or overview questions, and check the results, since below about 0.3 they quickly go off-topic.
- **`fetch_k` must be larger than `k`.** MMR can only choose among the `fetch_k` candidates. If `fetch_k` is too close to `k` there is nothing to choose; if it's too large, weak candidates enter the pool. 4–5× `k` is a common choice.
- **Small datasets hide the effect.** With 10 chunks, `fetch_k=20` puts the whole file in the pool. The difference between similarity and MMR is much bigger on large, repetitive data.
- **Chunks mix topics.** Each chunk here holds two unrelated sentences (e.g. agents + RAG pipeline), so picking a chunk for one topic also brings in the other. Smaller or [semantic chunks](../Semantic-Chunking/readme.md) make MMR's choices cleaner.
- **The prompt does not restrict the LLM to the context.** "Answer the question based on the context provided" is loose. The answer mentions agent types like `ReAct` and `ZeroShot` and nested agents, none of which are in the retrieved chunks. Adding *"Use only the context. If the answer is not in it, say you don't know."* reduces this.
- **MMR order is selection order.** The second document is "the best next pick", not necessarily the second most relevant. If order matters to the LLM, add a [reranker](../Reranking/readme.md) after MMR.
- **Unused code.** `ChatOpenAI` and `Document` are imported but not used.

---

## 7. Setup

From the project root:

```bash
pip install -r requirements.txt
```

The packages this folder needs are `langchain`, `langchain-classic`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-groq`, `langchain-openai` (only for the unused import in Cell 0), `sentence-transformers`, `faiss-cpu` and `python-dotenv`.

Create a `.env` file in the project root with your Groq key (free at [console.groq.com](https://console.groq.com)):

```
groq_api=your_groq_key_here
```

Then open `code.ipynb` **from inside the `MMR/` folder** (so `langchain_rag_dataset.txt` is found) and run the cells in order. Cells 2–4 need no API key; Cell 7 is the only cell that calls Groq. The first run downloads the embedding model (about 90 MB).

---

## 8. Quick Glossary

| Term | Meaning |
|---|---|
| **MMR (Maximal Marginal Relevance)** | Retrieval that picks chunks one at a time, balancing relevance to the question against similarity to chunks already picked |
| **Redundancy** | Retrieved chunks that repeat the same information |
| **Marginal relevance** | How much *new* relevant information a chunk adds on top of those already chosen |
| **k** | How many chunks the retriever finally returns (3 in this notebook) |
| **fetch_k** | How many similarity candidates MMR chooses from (default 20) |
| **lambda_mult (λ)** | 1.0 = pure relevance, 0.0 = pure diversity (default 0.5) |
| **Similarity search** | Returns the k chunks closest to the question, each scored independently |
| **FAISS** | Facebook AI Similarity Search, a fast in-memory vector index |
| **Retriever** | An object that takes a question and returns relevant chunks |
| **Stuff chain** | A chain that puts all retrieved chunks into one prompt |
| **Retrieval chain** | A chain that runs the retriever, then the documents chain, and returns `input`, `context` and `answer` |
| **`init_chat_model`** | LangChain helper that creates a chat model from a `"provider:model"` string |
| **Groq** | A cloud provider that serves open-weight LLMs very fast, with a free tier |
