# Reranking for RAG: Asking an LLM to Put the Best Chunks First

This folder covers **reranking**, a second pass over the chunks a retriever returns that puts them into a better order before they reach the LLM. The previous folders (see [Semantic-Chunking](../Semantic-Chunking/readme.md) and [Hybrid-Retriever](../Hybrid-Retriever/readme.md)) focused on *finding* candidate chunks. A retriever is built to be fast, so its ranking is only approximate. Reranking spends a little more compute on a small number of candidates to get the order right.

The notebook uses the simplest kind of reranker, an **LLM reranker**: the retrieved chunks are numbered, pasted into a prompt with the question, and the LLM replies with the numbers in order of relevance.

The notebook builds the idea up in four steps:

1. **Load and chunk a text file**: `TextLoader` plus `RecursiveCharacterTextSplitter` turn `langchain_sample.txt` into 6 chunks.
2. **First-stage retrieval**: a FAISS vector store with `all-MiniLM-L6-v2` embeddings returns the top candidates for the question.
3. **LLM reranking**: a ranking prompt, a Groq-hosted LLM and `StrOutputParser` return a string like `1,2,4,3,5,6`.
4. **Parse and reorder**: the string is turned into list indexes and used to reorder the retrieved `Document`s.

```
Reranking/
├── code.ipynb              # Notebook with retrieval + LLM reranking
├── langchain_sample.txt    # 10 sentences about LangChain, RAG, BM25, FAISS, agents, memory
└── readme.md               # This file
```

---

## 1. Core Concepts

### Two-stage retrieval
**Definition:** Splitting search into a fast, rough first stage and a slower, more precise second stage.

```
Question
   │
   ▼
Stage 1: Retriever (FAISS / BM25 / hybrid)      fast, searches ALL chunks, rough order
   │   top-k candidates (e.g. 8–50)
   ▼
Stage 2: Reranker (LLM / cross-encoder)          slow, looks at FEW chunks, precise order
   │   best few (e.g. top 3)
   ▼
LLM writes the answer from the best chunks
```

**Why two stages:** the precise model is too slow to run on every chunk in the database, and the fast model is not precise enough to trust its order. Running the fast one first and the precise one on its shortlist gives most of the precision at a small cost.

### Why the retriever's order is not good enough
A dense retriever (see the [Embeddings](../Embeddings/readme.md) folder) embeds the question and each chunk **separately**, then compares the two vectors with cosine similarity. Each chunk is squeezed into one vector of 384 numbers before the question is even known, so details get lost:

- A chunk that mentions many topics gets an "average" vector that is moderately close to many questions.
- The question *"How do I use memory and tools?"* and a chunk that only says *"LangChain has components for memory"* can score high because they share a topic, even though the chunk doesn't explain how.
- The retriever has no idea which of two similar chunks actually **answers** the question.

### Reranking
**Definition:** Re-scoring a short list of retrieved chunks with a model that reads the **question and each chunk together**, then sorting them by that new score.

Because the reranker sees both texts at the same time, it can judge "does this chunk answer this question?" instead of "are these two vectors close?".

### Kinds of rerankers

| Reranker | How it scores | Speed / cost | Notes |
|---|---|---|---|
| **LLM reranker** (this notebook) | One prompt with all candidates; the LLM returns a ranked list | Slow, costs tokens | No extra model to install; quality depends on the LLM and the prompt; output must be parsed |
| **Cross-encoder** (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-base`) | Runs question + chunk through a small model, outputs one relevance score per pair | Fast, runs locally | The most common choice in production |
| **Hosted rerank APIs** (e.g. Cohere Rerank, Jina Reranker) | Same idea as a cross-encoder, served over an API | Fast, paid per call | Plugs into LangChain via `ContextualCompressionRetriever` |
| **Reciprocal Rank Fusion** (see [Hybrid-Retriever](../Hybrid-Retriever/readme.md)) | Merges rankings from several retrievers by position only | Instant | Combines lists, does not read the text |

### Bi-encoder vs cross-encoder (the idea behind rerankers)

| | Bi-encoder (the retriever) | Cross-encoder / LLM (the reranker) |
|---|---|---|
| Input | Question and chunk embedded **separately** | Question and chunk read **together** |
| Can pre-compute chunk side? | Yes, stored in the vector DB | No, must run at query time |
| Speed on 1 million chunks | Milliseconds | Hours, so it's only used on the top-k |
| Precision | Good | Better |

---

## 2. Use Cases

Reranking pays off when the **top few** chunks matter more than the rest:

- **Small context budgets**: if only 3 chunks fit in the prompt (or you pay per token), the 3 you send should be the best 3, not just 3 close ones.
- **Questions with near-duplicate candidates**: product docs, policies and FAQs often have several chunks that look alike. The reranker picks the one that actually answers the question.
- **After hybrid retrieval**: BM25 and dense search together return a wide, noisy list. A reranker cleans it up.
- **Multi-part questions**: *"How do I use memory and tools?"* needs a chunk that covers both. A reranker that reads the question can notice that.
- **Search and recommendation UIs**: anywhere a user sees a ranked list, the first result matters most.

It is **less useful** when:
- You only retrieve 1–2 chunks, so there is nothing to reorder.
- Latency is critical. An LLM reranker adds a full LLM call (often 0.5–3 seconds) before the answer call.
- The retriever already returns the right chunk first almost every time, as on very small or very clean datasets.

---

## 3. Libraries Used

| Library | Purpose in this notebook |
|---|---|
| `langchain-community` | `TextLoader` to read the `.txt` file and the `FAISS` vector store wrapper |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` to cut the text into chunks |
| `langchain-huggingface` | `HuggingFaceEmbeddings`, the LangChain wrapper around sentence-transformers |
| `sentence-transformers` | Downloads and runs the `all-MiniLM-L6-v2` embedding model locally |
| `faiss-cpu` | The vector index that does the fast similarity search |
| `langchain-core` | `PromptTemplate`, `StrOutputParser`, `Document` |
| `langchain` | `init_chat_model`, a single function that creates a chat model from a `"provider:model"` string |
| `langchain-groq` | `ChatGroq`, the chat model client for Groq's hosted LLMs |
| `python-dotenv` | Loads the Groq API key from a `.env` file |
| `langchain-openai` | Imported (`ChatOpenAI`) but not used; the notebook uses Groq instead |

---

## 4. The Code, Cell by Cell

### Cell 0: Imports

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI   # or from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
```

| Import | What it is |
|---|---|
| `TextLoader` | Reads a plain text file into a list with one `Document`. |
| `RecursiveCharacterTextSplitter` | Splits text into chunks of a maximum size, preferring to break at paragraphs, then lines, then spaces. |
| `ChatOpenAI` | The OpenAI chat client. Imported but **not used**; the comment shows that `ChatGroq` can take its place. |
| `PromptTemplate` | A reusable prompt with `{placeholders}` that get filled in at run time. |
| `Document` | LangChain's text container: `page_content` plus `metadata`. Imported but not used directly; the loader and retriever create the `Document`s. |
| `StrOutputParser` | Takes the LLM's message object and returns just its text as a plain `str`. |

The `DeprecationWarning` in the output says `langchain-community` is being sunset. The code still works; in new projects, prefer the standalone integration packages.

---

### Cell 1: Load and chunk the text file

```python
loader=TextLoader("langchain_sample.txt")
raw_docs=loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)
docs
```

| Line | What it does |
|---|---|
| `TextLoader("langchain_sample.txt")` | Points at the sample file. The path is relative, so run the notebook from inside `Reranking/`. |
| `loader.load()` | Returns a list with one `Document` holding the whole file, with `metadata={"source": "langchain_sample.txt"}`. |
| `chunk_size=500` | Each chunk is at most 500 characters. |
| `chunk_overlap=50` | Up to 50 characters may repeat between neighbouring chunks, so a sentence cut at a boundary still appears whole somewhere. |
| `splitter.split_documents(raw_docs)` | Splits each document and copies its metadata to every chunk. |

The sample file has 10 sentences (one per line), each about 170–250 characters. The splitter packs lines together until the next one would go past 500 characters, which gives **6 chunks**:

| # | Chunk content (short) |
|---|---|
| A | LangChain is a flexible framework… components for prompt management, chains, memory, and agents. |
| B | LangChain integrates with OpenAI, Hugging Face, and Cohere… |
| C | RAG retrieves external knowledge… + BM25 is a sparse keyword method… |
| D | Dense retrieval uses embeddings… + LangChain supports hybrid retrieval… |
| E | FAISS is a library for nearest neighbour search… + **Agents** in LangChain decide which **tools** to use… |
| F | LangChain supports **tool integration**… + **Memory** in LangChain enables context retention… |

The letters are just labels for this readme, so the chunks can be followed through the next cells.

---

### Cell 2: The question

```python
query="How can i use langchain to build an application with memory and tools?"
```
A two-part question: it needs both **memory** and **tools**. The ideal answer chunk is F, which covers both. Chunks A and E each mention one of them.

---

### Cell 3: Vector store and retriever

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore=FAISS.from_documents(docs,embedding_model)
retriever=vectorstore.as_retriever(search_kwargs={"k":8})
```

| Line | What it does |
|---|---|
| `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` | Loads the small, free embedding model (384-number vectors, runs on CPU). The first run downloads about 90 MB. |
| `FAISS.from_documents(docs, embedding_model)` | Embeds all 6 chunks and builds an in-memory FAISS index, keeping the text and metadata next to each vector. |
| `as_retriever(search_kwargs={"k":8})` | Wraps the store in a retriever that returns the **8** closest chunks. There are only 6 chunks, so it returns **all 6**, just sorted by similarity. |

`k=8` is deliberate in a reranking setup: the first stage should return **more** candidates than you finally need, so the reranker has something to choose from. On a real dataset you would retrieve 20–50 and keep the top 3–5 after reranking.

The `Loading weights: 0/103` progress bar and the `HF_TOKEN` warning are normal. The warning only means downloads are rate-limited without a free HuggingFace token.

---

### Cell 4: A second setup with `load_dotenv` and `ChatGroq`

```python
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore_hf = FAISS.from_documents(docs, embeddings)
retriever_hf = vectorstore_hf.as_retriever(search_kwargs={"k": 8})

llm = ChatGroq(
    model="gemma-7b-it",
    api_key=os.getenv("groq_api")
)
```

- `load_dotenv()` reads the `.env` file (in the current or a parent folder) and puts its `KEY=value` lines into the environment. Cell 7 depends on this.
- `retriever_hf` is a **copy** of Cell 3's retriever. `"all-MiniLM-L6-v2"` and `"sentence-transformers/all-MiniLM-L6-v2"` are the same model, so the two retrievers return the same results. The rest of the notebook uses `retriever` from Cell 3.
- `ChatGroq(model="gemma-7b-it", ...)` creates an LLM client. Groq has since retired `gemma-7b-it`, so calling this model would fail. It is never called, because Cell 7 replaces `llm` with a current model. Creating the client does not contact Groq, which is why the cell runs without error.

---

### Cells 5 and 6: Inspect the retrievers

```python
retriever
retriever_hf
```
Both print a `VectorStoreRetriever` with `tags=['FAISS', 'HuggingFaceEmbeddings']` and `search_kwargs={'k': 8}`. The only difference is the memory address of the FAISS object, which confirms they are two separate indexes built the same way.

---

### Cell 7: The LLM used for reranking

```python
from langchain.chat_models import init_chat_model
os.environ["GROQ_API_KEY"]=os.getenv("groq_api")
llm=init_chat_model("groq:openai/gpt-oss-20b")
llm
```

| Line | What it does |
|---|---|
| `os.environ["GROQ_API_KEY"]=os.getenv("groq_api")` | Copies the key from the `.env` name (`groq_api`) to the name `ChatGroq` looks for (`GROQ_API_KEY`). |
| `init_chat_model("groq:openai/gpt-oss-20b")` | A provider-neutral factory. The part before `:` picks the provider (`groq` → `ChatGroq`), and the rest is the model name. Switching to another provider only means changing this string, e.g. `"openai:gpt-4o-mini"`. |

The printed `ChatGroq` object shows the model's profile: `openai/gpt-oss-20b` is OpenAI's open-weight 20B model hosted on Groq, with a 131,072-token input window and **`reasoning_output: True`**. It is a reasoning model, so it "thinks" before answering, which helps with a ranking task but makes each call slower.

---

### Cell 8: The ranking prompt

```python
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Your task is to rank the following documents from most to least relevant to the user's question.

User Question: "{question}"

Documents:
{documents}

Instructions:
- Think about the relevance of each document to the user's question.
- Return a list of document indices in ranked order, starting from the most relevant.

Output format: comma-separated document indices (e.g., 2,1,3,0,...)
""")
```

This prompt **is** the reranker. Each part has a job:

| Part | Why it is there |
|---|---|
| Role and task | Tells the model it is ranking, not answering the question. |
| `{question}` | The user's question, in quotes so it stands apart from the instructions. |
| `{documents}` | The numbered list of candidate chunks (built in Cell 12). |
| "Think about the relevance…" | Asks the model to judge each chunk against the question. |
| "Output format: comma-separated document indices" | Forces a short, machine-readable answer that code can parse in Cell 16. Without it, the model would write a paragraph. |

`from_template` detects `question` and `documents` as the two input variables.

> **Watch out:** the example `2,1,3,0,...` contains a `0`, which hints at 0-based numbering, but Cell 12 numbers the documents from **1**. The model followed the list and answered 1-based here, but a model that copies the example could answer with a `0`. Cell 16 subtracts 1, so a `0` would become `-1` and be dropped by the filter in Cell 18. Making the example match the numbering, e.g. `2,1,3,4,...`, removes the ambiguity.

---

### Cells 9 and 10: First-stage retrieval

```python
retrieved_docs=retriever.invoke(query)
retrieved_docs
```
Embeds the question and returns the chunks sorted by vector similarity. With `k=8` and only 6 chunks, it returns all 6. This order is what the reranker will try to improve:

| Retriever rank | Chunk |
|---|---|
| 1 | F: tool integration + memory |
| 2 | A: LangChain is a flexible framework… memory, and agents |
| 3 | B: integrates with OpenAI, Hugging Face, Cohere |
| 4 | E: FAISS + agents decide which tools to use |
| 5 | C: RAG + BM25 |
| 6 | D: dense retrieval + hybrid retrieval |

The retriever already did well at the top: F (memory **and** tools) is first. But B, which is about third-party integrations, sits above E, which actually talks about agents using **tools**.

---

### Cell 11: The reranking chain

```python
chain=prompt| llm | StrOutputParser()
chain
```
An **LCEL** chain: the `|` operator passes each step's output to the next step.

```
{"question": ..., "documents": ...}
      ▼
prompt            fills the template → one prompt string
      ▼
llm               sends it to Groq → AIMessage("1,2,4,3,5,6")
      ▼
StrOutputParser   AIMessage → "1,2,4,3,5,6"
```

Printing `chain` shows the three steps joined with `|`.

---

### Cells 12 to 14: Number the documents for the prompt

```python
doc_lines = [f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)]
formatted_docs = "\n".join(doc_lines)
```

| Line | What it does |
|---|---|
| `enumerate(retrieved_docs)` | Gives each document with its position `i` (0, 1, 2, …). |
| `f"{i+1}. {doc.page_content}"` | Writes it as `"1. text"`, `"2. text"`, … The `+1` makes the numbering start at 1, which reads more naturally for the LLM. |
| `"\n".join(doc_lines)` | Glues them into one string, one document per line, to drop into `{documents}`. |

Cell 13 shows `doc_lines` as a list of 6 numbered strings, and Cell 14 shows `formatted_docs` as one long string. The numbers are the only way the LLM can refer back to a document, so they must stay in the same order as `retrieved_docs`.

---

### Cell 15: Ask the LLM to rank

```python
response=chain.invoke({"question":query,"documents":formatted_docs})
response
```
Output:
```
'1,2,4,3,5,6'
```
The model kept F and A at the top and **swapped positions 3 and 4**: it moved E (agents choosing tools) above B (third-party integrations). That is the more sensible order for a question about memory and tools, and it's the kind of judgement a vector score can't make.

---

### Cell 16: Parse the answer into list indexes

```python
indices = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
indices
```

| Piece | What it does |
|---|---|
| `response.split(",")` | `"1,2,4,3,5,6"` → `["1", "2", "4", "3", "5", "6"]` |
| `x.strip()` | Removes spaces, so `"1, 2, 4"` also works. |
| `if x.strip().isdigit()` | Skips anything that is not a plain number, such as `"Doc 3"` or an empty string from a trailing comma. |
| `int(...) - 1` | Converts the 1-based document numbers back to 0-based Python list indexes. |

Output:
```
[0, 1, 3, 2, 4, 5]
```

---

### Cells 17 and 18: Reorder the documents

```python
reranked_docs = [retrieved_docs[i] for i in indices if 0 <= i < len(retrieved_docs)]
reranked_docs
```
- Builds a new list by picking documents from `retrieved_docs` in the order the LLM chose.
- `if 0 <= i < len(retrieved_docs)` is a safety check: if the LLM invents a number that doesn't exist (like `9`, or `0` turned into `-1`), it is skipped instead of raising `IndexError` (or, for `-1`, silently picking the last document).

Cell 17 prints `retrieved_docs` again so it can be compared with the result. Only positions 3 and 4 differ.

---

### Cell 19: Print the final ranking

```python
print("\n📊 Final Reranked Results:")
for i, doc in enumerate(reranked_docs, 1):
    print(f"\nRank {i}:\n{doc.page_content}")
```
`enumerate(..., 1)` starts counting at 1, so the ranks print as `Rank 1` to `Rank 6`.

#### Before and after reranking

| Rank | Retriever (Cell 10) | After LLM rerank (Cell 19) |
|---|---|---|
| 1 | F: tool integration + memory | F: tool integration + memory |
| 2 | A: LangChain framework (memory, agents) | A: LangChain framework (memory, agents) |
| 3 | B: integrations with OpenAI, Hugging Face | **E: FAISS + agents choosing tools** ↑ |
| 4 | E: FAISS + agents choosing tools | **B: integrations with OpenAI, Hugging Face** ↓ |
| 5 | C: RAG + BM25 | C: RAG + BM25 |
| 6 | D: dense + hybrid retrieval | D: dense + hybrid retrieval |

On a 6-chunk file the change is small, because the retriever had little room to go wrong. On a real knowledge base with hundreds of similar chunks, the reranker's reordering matters much more.

---

## 5. Using the Reranked Chunks

The notebook stops at the reordered list. In a full RAG pipeline you would keep only the top few and send them to the answer prompt:

```python
top_docs = reranked_docs[:3]
context = "\n\n".join(d.page_content for d in top_docs)

answer_prompt = PromptTemplate.from_template(
    "Answer using only this context:\n\n{context}\n\nQuestion: {question}"
)
answer = (answer_prompt | llm | StrOutputParser()).invoke(
    {"context": context, "question": query}
)
```

The same idea with a **cross-encoder** instead of an LLM, using LangChain's built-in pieces (needs `sentence-transformers`, already installed for the embeddings):

```python
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

rerank_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever,   # the k=8 FAISS retriever from Cell 3
)
top_docs = rerank_retriever.invoke(query)
```
The cross-encoder gives each chunk a numeric score, so there is no text to parse, and it runs locally in milliseconds. (In older LangChain versions these imports live in `langchain.retrievers` instead of `langchain_classic.retrievers`.)

---

## 6. Limitations and Tips

- **Fragile output parsing.** The LLM may answer `"Ranking: 1, 2, 4"`, `"[1, 2, 4]"` or add an explanation. The `isdigit()` filter handles some of this, but `"[1"` is skipped silently. Asking for JSON with structured output (`llm.with_structured_output(...)`) is more robust.
- **Missing or repeated numbers.** If the model leaves out a document, it disappears from `reranked_docs`. If it repeats one, that document appears twice. Remove duplicates and append any missing documents at the end to be safe.
- **Extra latency and cost.** The reranker is a whole LLM call before the answer call, and a reasoning model like `gpt-oss-20b` is slower still. Every candidate chunk is sent as input tokens.
- **Long candidate lists.** LLMs pay less attention to the middle of long prompts. With 30+ candidates, rank in batches or use a cross-encoder.
- **Not deterministic.** The same input can give a slightly different order on another run. Set `temperature=0` for more repeatable rankings.
- **Reranking can't add what retrieval missed.** It only reorders the candidates. If the right chunk isn't in the top-k, no reranker can find it, so retrieve generously (k = 20–50) in stage 1.
- **Retrieving `k` larger than the data.** Here `k=8` with 6 chunks returns everything. That's fine for a demo, but on real data `k` should be much smaller than the number of chunks, and you should keep only the reranker's top few.
- **Unused code.** `ChatOpenAI`, `Document`, `retriever_hf` and the `gemma-7b-it` client in Cell 4 are not used by the reranking flow and can be removed.

---

## 7. Setup

From the project root:

```bash
pip install -r requirements.txt
```

The packages this folder needs are `langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-groq`, `sentence-transformers`, `faiss-cpu` and `python-dotenv`.

Create a `.env` file in the project root with your Groq key (free at [console.groq.com](https://console.groq.com)):

```
groq_api=your_groq_key_here
```

Then open `code.ipynb` **from inside the `Reranking/` folder** (so `langchain_sample.txt` is found) and run the cells in order. Cell 4 calls `load_dotenv()`, which Cell 7 needs, so don't skip it. Cell 4 also imports `ChatGroq`, which Cell 0 does not. Cells 0–3 and 9–10 need no API key; Cell 15 is the only cell that calls Groq. The first run downloads the embedding model (about 90 MB); after that, everything except the LLM call works offline.

---

## 8. Quick Glossary

| Term | Meaning |
|---|---|
| **Reranking** | Re-scoring a short list of retrieved chunks with a more precise model and sorting them again |
| **Two-stage retrieval** | A fast retriever finds candidates, then a slower reranker orders them |
| **Candidate** | A chunk returned by the first-stage retriever, waiting to be reranked |
| **LLM reranker** | An LLM that reads the question and the numbered candidates and returns a ranked list |
| **Cross-encoder** | A model that reads question + chunk together and outputs one relevance score |
| **Bi-encoder** | A model that embeds question and chunk separately (what the retriever uses) |
| **k** | How many chunks the retriever returns (8 in this notebook) |
| **top_n** | How many chunks to keep after reranking |
| **FAISS** | Facebook AI Similarity Search, a fast in-memory vector index |
| **Retriever** | An object that takes a question and returns the most relevant chunks |
| **`init_chat_model`** | LangChain helper that creates a chat model from a `"provider:model"` string |
| **LCEL** | LangChain Expression Language; connects steps with `\|` into a chain |
| **`StrOutputParser`** | Turns the LLM's message object into a plain string |
| **0-based / 1-based** | Counting from 0 (Python lists) or from 1 (the numbers shown to the LLM) |
| **Groq** | A cloud provider that serves open-weight LLMs very fast, with a free tier |
| **Reasoning model** | An LLM that works through the problem internally before giving its final answer |
