# Hybrid Retriever for RAG: Combining Keyword Search and Semantic Search

This folder covers **hybrid retrieval**, which means running two kinds of search on the same documents and merging their results:

- **Dense retrieval** (semantic search): embeddings plus a FAISS vector store. It finds text with the same *meaning* as the question, even when the words are different.
- **Sparse retrieval** (keyword search): BM25. It finds text that contains the same *words* as the question, which matters for names, codes, acronyms and rare terms.

Each one catches things the other misses, so combining them usually gives better retrieval than either alone. The previous folders (see [LangChain-FAISS](../LangChain-FAISS/) and [Semantic-Chunking](../Semantic-Chunking/readme.md)) used dense retrieval only.

The notebook builds the idea up in three steps:

1. **Two retrievers on the same documents**: a FAISS dense retriever and a BM25 sparse retriever.
2. **One hybrid retriever**: LangChain's `EnsembleRetriever` merges the two ranked lists with weighted **Reciprocal Rank Fusion** and removes duplicates.
3. **A full RAG chain on top**: the hybrid retriever feeds a Groq-hosted LLM through `create_retrieval_chain`.

```
Hybrid-Retriever/
├── code.ipynb   # Notebook with the hybrid retriever and RAG chain
└── readme.md    # This file
```

---

## 1. Core Concepts

### Dense retrieval (semantic search)
**Definition:** Turn every document and the question into embedding vectors, then return the documents whose vectors are closest to the question's vector.

- **Good at:** synonyms and paraphrases. *"How do I build an app with LLMs?"* matches *"LangChain helps build LLM applications"* even though *app* ≠ *applications*.
- **Weak at:** exact tokens the embedding model has never seen or blurs together, such as product codes (`XJ-4402`), error codes, people's names and new jargon.

### Sparse retrieval (keyword search) and BM25
**Definition:** Represent text by the words it contains and score documents by how well their words match the question's words. It is called *sparse* because each document's vector has one slot per vocabulary word, and almost all of them are zero.

**BM25** ("Best Matching 25") is the standard keyword-scoring formula used by search engines such as Elasticsearch and Lucene. For each query word that appears in a document, it adds points based on:

| Factor | Effect |
|---|---|
| **Term frequency** | More occurrences of the word in the document → higher score, but with diminishing returns (the 10th occurrence adds much less than the 1st) |
| **Inverse document frequency (IDF)** | Rare words count more than common ones. Matching *"Pinecone"* is worth more than matching *"the"* |
| **Document length** | Long documents are penalised slightly, so they don't win just by containing more words |

- **Good at:** exact matches, rare terms, names, IDs. Needs no model and no GPU.
- **Weak at:** synonyms and meaning. *"car"* does not match *"automobile"*, and *"LLMs"* does not match *"LLM"*.

### Why combine them
| Query | Dense retrieval | BM25 |
|---|---|---|
| *"How do I make my app remember past chats?"* | ✅ finds "conversation memory" docs | ❌ no shared words |
| *"What does error E-1043 mean?"* | ❌ the code is noise to the embedding | ✅ exact token match |
| *"Pinecone pricing"* | ✅ | ✅ |

A hybrid retriever runs both searches and merges the results, so a document only has to be found by one of them to be returned.

### Reciprocal Rank Fusion (RRF)
**The problem:** the two retrievers score on different scales. FAISS returns an L2 distance (lower is better, e.g. `0.73`), while BM25 returns an unbounded relevance score (higher is better, e.g. `1.26`). You cannot simply add them.

**The fix:** ignore the raw scores and use only each document's **rank** (1st, 2nd, 3rd…) in each list:

```
RRF score(doc) = Σ over retrievers   weight_i / (rank_i(doc) + c)
```

- `rank_i(doc)` is the document's position in retriever *i*'s list (1 = top). If a retriever didn't return the document, it adds nothing.
- `c` is a smoothing constant, `60` by default. It stops the 1st place from completely dominating the 2nd and 3rd.
- `weight_i` is how much you trust retriever *i*. This notebook uses `0.7` for dense and `0.3` for BM25.

Documents found by **both** retrievers get points twice, so they rise to the top. The final list is sorted by RRF score, with duplicates removed.

### `EnsembleRetriever` vs `MergerRetriever`
LangChain has two classes that combine retrievers, and they behave very differently:

| | `EnsembleRetriever` (used here) | `MergerRetriever` |
|---|---|---|
| How it merges | Weighted RRF, sorted by fused score | Interleaves the lists: 1st of A, 1st of B, 2nd of A, 2nd of B… |
| Weights | `weights=[0.7, 0.3]` | Not supported (an extra `weight=` argument is silently ignored) |
| Duplicates | Removed | Kept, so the same document appears twice |
| Import | `from langchain_classic.retrievers import EnsembleRetriever` | `from langchain_classic.retrievers import MergerRetriever` |

An earlier version of this notebook used `MergerRetriever`, which returned 7 results with every LangChain document listed twice. Switching to `EnsembleRetriever` fixed it.

---

## 2. Use Cases

- **Technical documentation and support bots**: questions mix natural language (*"why does login fail"*) with exact identifiers (error codes, function names, config keys).
- **E-commerce and catalogue search**: SKUs and model numbers need exact matching; descriptions need semantic matching.
- **Legal, medical and financial documents**: specific terms, clause numbers and drug names must match exactly.
- **Domain-specific corpora**: when your jargon is not well covered by a general embedding model, BM25 makes up for it.
- **As a default for production RAG**: hybrid search is a cheap, reliable improvement over dense-only retrieval, and most vector databases (Pinecone, Weaviate, Qdrant, Elasticsearch) support it natively.

---

## 3. Libraries Used

| Library | Used for |
|---|---|
| `langchain-core` | `Document`, `PromptTemplate` |
| `langchain-community` | `FAISS` vector store wrapper and `BM25Retriever` |
| `langchain-huggingface` | `HuggingFaceEmbeddings` (runs `all-MiniLM-L6-v2` locally) |
| `langchain-classic` | `EnsembleRetriever`, `create_stuff_documents_chain`, `create_retrieval_chain` (these moved here from `langchain` in LangChain 1.x) |
| `langchain-groq` | `ChatGroq`, the LLM client |
| `faiss-cpu` | The vector index behind the dense retriever |
| `rank_bm25` | The BM25 implementation behind `BM25Retriever` |
| `sentence-transformers` | Loads the embedding model |
| `python-dotenv` | Loads the Groq API key from `.env` |

---

## 4. The Code, Cell by Cell

### Cell 0: Imports
```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
```
Importing `langchain_community` prints a `DeprecationWarning` saying the package is being sunset. It still works, so you can ignore the warning.

### Cell 1: Sample documents
```python
docs = [
    Document(page_content="LangChain helps build LLM applications."),
    Document(page_content="Pinecone is a vector database for semantic search."),
    Document(page_content="The Eiffel Tower is located in Paris."),
    Document(page_content="Langchain can be used to develop agentic ai application."),
    Document(page_content="Langchain has many types of retrievers.")
]
```
Five one-sentence documents: three about LangChain, one about a vector database, and one unrelated "distractor" (the Eiffel Tower) that should never be retrieved for an LLM question.

### Cell 2: Dense retriever
```python
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
dense_vectorstore = FAISS.from_documents(docs, embedding_model)
dense_retriever = dense_vectorstore.as_retriever()
```
1. Load the `all-MiniLM-L6-v2` embedding model (384-dimensional vectors, runs on CPU).
2. Embed all five documents and store them in an in-memory FAISS index.
3. Wrap the index as a retriever. It returns the top **k = 4** documents by default.

### Cell 3: Sparse retriever and the hybrid retriever
```python
sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 3

hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]
)
```
- `BM25Retriever.from_documents` builds the BM25 index straight from the text. There is no model and no embedding step.
- `k = 3` makes BM25 return its top 3 documents.
- `EnsembleRetriever` runs both retrievers on every query and fuses the lists with weighted RRF. The weights `[0.7, 0.3]` match the order of `retrievers`, so dense search counts for 70% and BM25 for 30%.

### Cell 4: Inspect the retriever
Displays the `EnsembleRetriever` object, which shows its two child retrievers and weights.

### Cell 5: Query the hybrid retriever
```python
query = "How can I build an application using LLMs?"
results = hybrid_retriever.invoke(query)
```

#### What each retriever returns on its own

**Dense (FAISS, L2 distance, lower = closer):**

| Rank | Distance | Document |
|---|---|---|
| 1 | 0.729 | LangChain helps build LLM applications. |
| 2 | 1.526 | Langchain can be used to develop agentic ai application. |
| 3 | 1.707 | Pinecone is a vector database for semantic search. |
| 4 | 1.827 | Langchain has many types of retrievers. |

**Sparse (BM25, higher = better):**

| Rank | Score | Document | Why |
|---|---|---|---|
| 1 | 1.26 | LangChain helps build LLM applications. | Shares the word `build` |
| 2 | 0.97 | Langchain can be used to develop agentic ai application. | Shares the word `can` |
| 3 | 0.00 | Langchain has many types of retrievers. | No overlap; it only fills the 3rd slot |

This shows BM25's weakness clearly. `BM25Retriever` splits text on spaces and does no further cleanup, so `LLMs?` ≠ `LLM` and `application` ≠ `application.`. Document 2 scored only because of the common word *can*. Dense search understood the meaning without any exact matches.

#### How RRF fuses them (`c = 60`)

| Document | Dense part | BM25 part | RRF score | Final rank |
|---|---|---|---|---|
| LangChain helps build LLM applications. | 0.7 / (1+60) | 0.3 / (1+60) | **0.01639** | 1 |
| Langchain can be used to develop agentic ai application. | 0.7 / (2+60) | 0.3 / (2+60) | **0.01613** | 2 |
| Langchain has many types of retrievers. | 0.7 / (4+60) | 0.3 / (3+60) | **0.01570** | 3 |
| Pinecone is a vector database for semantic search. | 0.7 / (3+60) | not returned | **0.01111** | 4 |

Output:
```
🔹 Document 1:
LangChain helps build LLM applications.

🔹 Document 2:
Langchain can be used to develop agentic ai application.

🔹 Document 3:
Langchain has many types of retrievers.

🔹 Document 4:
Pinecone is a vector database for semantic search.
```
Notice that *"Langchain has many types of retrievers"* was 4th in dense search but finishes 3rd overall, ahead of Pinecone, because BM25 also returned it. Being found by both retrievers counts for more than a slightly better rank in one of them. The Eiffel Tower document is never returned.

### Cell 6: Load the API key
```python
import os
from dotenv import load_dotenv
load_dotenv()
```
Reads the `.env` file in the project root so `os.environ["groq_api"]` is available.

### Cell 7: Imports for the RAG chain
```python
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
```

### Cell 8: Prompt and LLM
```python
prompt = PromptTemplate.from_template("""
Answer the question based on the context below.

Context:
{context}

Question: {input}
""")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ["groq_api"],
    temperature=0.2
)
```
- The prompt has two placeholders. `{context}` is filled with the retrieved documents and `{input}` with the user's question. These names are required by the chain helpers in the next cell.
- `ChatGroq` calls the open-weight `gpt-oss-20b` model hosted on Groq. A low `temperature` (0.2) keeps answers focused and repeatable.

### Cell 9: Build the RAG chain
```python
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=hybrid_retriever, combine_docs_chain=document_chain)
```
- `create_stuff_documents_chain` "stuffs" all retrieved documents into the `{context}` slot of the prompt, then calls the LLM.
- `create_retrieval_chain` puts the retriever in front: question → hybrid retriever → document chain → answer.

```
{"input": question}
        │
        ▼
 hybrid_retriever ──► dense (FAISS) ─┐
                  └─► sparse (BM25) ─┴─► RRF fuse ─► docs
        │
        ▼
 prompt(context=docs, input=question) ──► ChatGroq ──► answer
```

### Cell 10: Ask a question
```python
query = {"input": "How can I build an app using LLMs?"}
response = rag_chain.invoke(query)

print("Answer:", response["answer"])
print("Source Documents:", response["context"])
```
The chain returns a dict with three keys: `input` (the question), `context` (the documents the hybrid retriever found) and `answer` (the LLM's reply). Printing `context` lets you check which documents the answer was based on.

---

## 5. Tuning the Hybrid Retriever

| Knob | Where | Effect |
|---|---|---|
| `weights` | `EnsembleRetriever(weights=[dense, sparse])` | Shift towards `[0.5, 0.5]` or BM25-heavy if your queries contain many codes, names or exact terms. Shift towards dense if users paraphrase a lot. |
| `k` per retriever | `as_retriever(search_kwargs={"k": 5})`, `sparse_retriever.k = 5` | How many candidates each side contributes before fusion. More candidates = better recall, longer prompt. |
| `c` | `EnsembleRetriever(..., c=60)` | Lower values reward top ranks more strongly; higher values flatten the differences. `60` is the standard value from the original RRF paper. |
| BM25 tokenisation | `BM25Retriever.from_documents(docs, preprocess_func=...)` | Pass a function that lowercases and strips punctuation (or stems words) so `LLMs?` can match `LLM`. |

A good way to tune the weights is to collect 20–50 real questions with known correct documents, then try a few weight settings and count how often the right document lands in the top k.

---

## 6. Limitations and Tips

- **BM25's default tokeniser is very basic.** It only splits on spaces, so punctuation and plurals break matches (see Cell 5). Add a `preprocess_func` for real data.
- **BM25 always fills its k slots.** Like a dense retriever, it returns k documents even if some score zero. RRF will still give them some points. Keep `k` small for BM25, or filter out zero-score results.
- **The BM25 index lives in memory.** `BM25Retriever` rebuilds its index from the documents each time and is not saved with FAISS. For large corpora, use a search engine or vector database with built-in hybrid search.
- **Two searches per query.** Hybrid retrieval roughly doubles retrieval time. BM25 is very cheap, so in practice the embedding call dominates.
- **Weights affect ranking, not filtering.** A document returned by only one retriever is still included. `EnsembleRetriever` returns every unique document from both lists (up to 4 + 3 here).
- **Deduplication uses the page text.** Two documents with identical text but different metadata are merged into one. Pass `id_key="your_metadata_field"` to deduplicate by an ID instead.
- **Answers are only as good as the context.** Tell the LLM to answer only from the context and to say when it doesn't know.

---

## 7. Setup

From the project root:

```bash
pip install -r requirements.txt
```

The packages this folder needs are `langchain-core`, `langchain-community`, `langchain-classic`, `langchain-huggingface`, `langchain-groq`, `faiss-cpu`, `rank_bm25`, `sentence-transformers` and `python-dotenv`. If you get `ModuleNotFoundError: No module named 'langchain_community'`, run `pip install langchain-community` inside the project's virtual environment and restart the notebook kernel.

Create a `.env` file in the project root with your Groq key (free at [console.groq.com](https://console.groq.com)):

```
groq_api=your_groq_key_here
```

Then run the cells in order. Cells 0–5 need no API key. Cells 8–10 need the Groq key. The first run downloads the embedding model (about 90 MB); after that it works offline.

**Troubleshooting `PermissionDeniedError: 403 - Access denied. Please check your network settings.`** This error comes from Groq rejecting your network, not your key (a bad key gives a 401). It usually happens behind a VPN or proxy. Turn the VPN off or switch networks, then re-run the cell.

---

## 8. Quick Glossary

| Term | Meaning |
|---|---|
| **Hybrid retrieval** | Combining keyword (sparse) and semantic (dense) search results |
| **Dense retrieval** | Search by comparing embedding vectors; matches meaning |
| **Sparse retrieval** | Search by matching words; matches exact terms |
| **BM25** | The standard keyword-scoring formula, based on term frequency, word rarity and document length |
| **TF (term frequency)** | How often a word appears in a document |
| **IDF (inverse document frequency)** | How rare a word is across all documents; rare words score higher |
| **RRF (Reciprocal Rank Fusion)** | Merges ranked lists by summing `weight / (rank + c)` for each document |
| **`c`** | RRF smoothing constant (default 60) |
| **`EnsembleRetriever`** | LangChain retriever that fuses several retrievers with weighted RRF |
| **`MergerRetriever`** | LangChain retriever that interleaves results without weights or deduplication |
| **FAISS** | Facebook AI Similarity Search, a fast in-memory vector index |
| **L2 distance** | Straight-line distance between vectors; FAISS's default score (lower = closer) |
| **k** | How many documents a retriever returns |
| **Stuff documents chain** | A chain that inserts all retrieved documents into one prompt |
| **Retrieval chain** | A chain that runs a retriever, then passes its documents to a document chain |
| **Groq** | A cloud provider that serves open-weight LLMs very fast, with a free tier |
| **Temperature** | LLM randomness setting; low = focused and repeatable, high = varied |
