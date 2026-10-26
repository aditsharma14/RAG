# Query Expansion for RAG: Rewriting the Question Before Retrieval

This folder covers **query expansion**, a *query enhancement* technique: before the user's question reaches the retriever, an LLM rewrites it into a longer query with synonyms, technical terms and related context. Short or vague questions ("CrewAI agents?") give the embedding model very little to work with; an expanded query gives it more words that might match the way the documents are written.

The other folders in this project improve **what happens to the chunks** ([MMR](../MMR/readme.md) picks varied chunks, [Reranking](../Reranking/readme.md) re-orders them, [Hybrid-Retriever](../Hybrid-Retriever/) mixes keyword and vector search). Query expansion works on the **other side**: it improves the question.

The notebook builds the pipeline in five steps:

1. **Load and chunk a text file**: `TextLoader` plus `RecursiveCharacterTextSplitter` turn `langchain_crewai_dataset.txt` into 241 chunks.
2. **Vector store and retriever**: FAISS with `all-MiniLM-L6-v2` embeddings, searched with MMR (`k=5`).
3. **Query expansion chain**: `PromptTemplate | llm | StrOutputParser()` turns a question into an expanded query string.
4. **Answer chain**: `create_stuff_documents_chain` pastes the retrieved chunks into a prompt for a Groq-hosted LLM.
5. **RAG pipeline**: a `RunnableMap` sends the *expanded* query to the retriever and the *original* question to the answer prompt.

```
Query-Expansion/
├── code.ipynb                    # Notebook: LLM query expansion → FAISS MMR retriever → stuff-documents answer
├── langchain_crewai_dataset.txt  # 200 lines: 20 paragraphs about LangChain and CrewAI, each repeated 10 times (v1–v10)
└── readme.md                     # This file
```

---

## 1. Core Concepts

### The problem: the question and the documents use different words
Vector search compares the meaning of the question with the meaning of each chunk. That works well when the question is a full sentence, but real users type things like:

- `"CrewAI agents?"`: two words and a question mark
- `"Langchain memory"`: no hint whether they want the classes, the concept, or how to configure it
- `"LLM forgetting old messages"`: the documents call this *ConversationBufferMemory* and *token limits*

The fewer words in the question, the less the embedding has to match against, and the more likely a loosely related chunk wins.

### Query expansion
**Definition:** An extra step before retrieval in which an LLM rewrites the user's query into a richer one, adding synonyms, related technical terms and context, and the retriever searches with the rewritten query instead of the original.

```
User question                     Expanded query (LLM)                         Retriever
"What types of memory      →      "LangChain memory types OR memory      →     chunks about
 does LangChain support?"          implementations OR conversation memory       ConversationBufferMemory,
                                   OR LangChain memory classes OR ..."          ConversationSummaryMemory
```

The **original** question is still the one the final LLM answers. Only the retriever sees the expanded version.

### Where it sits in the pipeline

```
                ┌──────────────────────────┐
question ──┬──► │ query_expansion_chain    │ ──► expanded query ──► retriever ──► chunks ─┐
           │    │ (prompt | llm | parser)  │                                              │
           │    └──────────────────────────┘                                              ▼
           └──────────────────────────────────────────────────────────────────────► answer prompt ──► llm ──► answer
                                    (original question)
```

### Query enhancement family
Query expansion is one of several ways to improve the question before retrieval:

| Technique | What the LLM produces | Retriever calls |
|---|---|---|
| **Query expansion** (this folder) | One longer query with synonyms and related terms | 1 |
| Query rewriting | One cleaner, better-phrased query (fixes typos, removes chit-chat) | 1 |
| Multi-query | Several different versions of the question; results are merged | several |
| Query decomposition | Several sub-questions for a multi-part question | several |
| HyDE | A fake *answer* paragraph, which is embedded instead of the question | 1 |

### Expansion vs no expansion

| | Original query | Expanded query |
|---|---|---|
| Extra cost | None | One extra LLM call per question (latency + tokens) |
| Short / vague questions | Weak matches | More terms to match against |
| Already precise questions | Works well | Little gain; can drift off-topic |
| Risk | Missing relevant chunks (low recall) | The LLM adds wrong meanings (e.g. airline crew scheduling for "CrewAI") |

---

## 2. Use Cases

Query expansion helps when the **question is weaker than the documents**:

- **Short or keyword-style questions**: search boxes and chatbots get two- or three-word queries. Expansion turns them into something closer to a sentence.
- **Vocabulary mismatch**: users say "remember the chat", the docs say "ConversationBufferMemory". Synonyms and technical terms bridge the gap.
- **Domain jargon and acronyms**: expanding "RAG" to "retrieval-augmented generation" or "k8s" to "Kubernetes" helps both vector and keyword search.
- **Keyword retrievers (BM25)**: keyword search only finds exact words, so added synonyms matter even more there than in vector search. It pairs well with a [hybrid retriever](../Hybrid-Retriever/).
- **Recall-focused search**: legal, research or support search, where missing a relevant document is worse than reading an extra one.

It is **less useful** when:
- The question is already long and precise. Expansion adds noise to a query that was fine.
- The name is ambiguous or unknown to the LLM. The model guesses a meaning, and a wrong guess pulls in the wrong chunks (see the CrewAI example in Section 5).
- Latency matters a lot. Every question now waits for an extra LLM call before retrieval starts.

---

## 3. Libraries Used

| Library | Purpose in this notebook |
|---|---|
| `langchain-community` | `TextLoader` to read the `.txt` file and the `FAISS` vector store wrapper |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` to cut the text into chunks |
| `langchain-huggingface` | `HuggingFaceEmbeddings`, the LangChain wrapper around sentence-transformers |
| `sentence-transformers` | Downloads and runs the `all-MiniLM-L6-v2` embedding model locally |
| `faiss-cpu` | The vector index that does the fast similarity / MMR search |
| `langchain-core` | `PromptTemplate`, `StrOutputParser`, `RunnableMap` and the `|` (pipe) operator |
| `langchain-classic` | `create_stuff_documents_chain` (moved here from `langchain` in LangChain 1.x) |
| `langchain` | `init_chat_model`, which creates a chat model from a `"provider:model"` string |
| `langchain-groq` | `ChatGroq`, created behind the scenes by `init_chat_model("groq:...")` |
| `python-dotenv` | Loads the Groq API key from a `.env` file |

---

## 4. The Code, Cell by Cell

### Cell 0: Title (Markdown)
A short description of the idea: rewrite a short or vague query into a richer one before retrieval. *Better query → better retrieved chunks → better final answer.*

---

### Cell 1: Imports

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
```

| Import | What it is |
|---|---|
| `TextLoader` | Reads a plain text file into a list with one `Document`. |
| `RecursiveCharacterTextSplitter` | Splits text into chunks of a maximum size, preferring to break at paragraphs, then lines, then spaces. |
| `HuggingFaceEmbeddings` | Turns text into vectors with a local sentence-transformers model. |
| `FAISS` | LangChain's wrapper around the FAISS index. |
| `init_chat_model` | Creates a chat model from a `"provider:model"` string. |
| `PromptTemplate` | A reusable prompt with `{placeholders}` filled in at run time. |
| `create_stuff_documents_chain` | Builds a chain that "stuffs" a list of documents into one prompt variable and calls the LLM. |
| `StrOutputParser` | Turns the LLM's message object into a plain Python string. |
| `RunnableMap` | Runs several functions on the same input and returns their results as a dict. Used to build `{"input": ..., "context": ...}`. |

---

### Cell 2: Load and chunk the text file

```python
loader=TextLoader("langchain_crewai_dataset.txt",encoding="utf-8")
raw_docs=loader.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
chunks=splitter.split_documents(raw_docs)
```

| Line | What it does |
|---|---|
| `TextLoader(..., encoding="utf-8")` | The file contains a curly apostrophe (`LangChain’s`). Without `encoding="utf-8"`, Windows reads it with its default code page and may fail. The path is relative, so run the notebook from inside `Query-Expansion/`. |
| `chunk_size=300` | Each chunk is at most 300 characters. |
| `chunk_overlap=50` | When a paragraph is longer than 300 characters and has to be cut, up to 50 characters are repeated in the next chunk so a sentence is not lost at the cut. |

#### About the dataset
The file has **200 lines**, but only **20 different paragraphs**: 10 about LangChain (chains, vector databases, hybrid retrieval, agents, memory, prompts, providers, modularity) and 10 about CrewAI (crews, roles, context-sharing, YAML configuration, scaling). Each paragraph appears **10 times**, tagged `(v1)` to `(v10)`.

---

### Cell 3: Inspect the chunks

```python
print(len(chunks))
chunks[:3]
```

Output: **241 chunks**. Most paragraphs fit in one 300-character chunk; longer ones are cut in two. For example, the first paragraph becomes:

```
chunk 1: "LangChain is an open-source framework ... memory, and agent orchestration. Developers can use"
chunk 2: "and agent orchestration. Developers can use LangChain to create end-to-end pipelines ... (v1)"
          └──────── 50-character overlap ────────┘
```

---

### Cell 4: Vector store and retriever

```python
embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore=FAISS.from_documents(chunks,embedding_model)
retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":5})
```

| Line | What it does |
|---|---|
| `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` | Loads the small, free embedding model (384-number vectors, runs on CPU). The first run downloads about 90 MB. |
| `FAISS.from_documents(...)` | Embeds all 241 chunks and builds an in-memory index. |
| `search_type="mmr"` | Uses [Maximal Marginal Relevance](../MMR/readme.md) instead of plain similarity, to try to avoid returning near-duplicates. |
| `search_kwargs={"k":5}` | Return 5 chunks. `fetch_k` (20) and `lambda_mult` (0.5) keep their defaults. |

The `Loading weights` progress bar and the `unauthenticated requests to the HF Hub` warning are normal. Setting an `HF_TOKEN` only gives faster downloads.

---

### Cell 5: Show the retriever

```python
retriever
```

Prints the retriever's settings: `search_type='mmr', search_kwargs={'k': 5}`.

---

### Cell 6: Retrieval with the plain query (baseline)

```python
retriever.invoke("LangChain memory")
```

This is the retriever on its own, with **no expansion**. It returns 5 chunks:

| # | Chunk |
|---|---|
| 1 | memory modules (ConversationBufferMemory / ConversationSummaryMemory) **(v10)** |
| 2 | LangChain workflows are modular… retrievers, memories, agents **(v9)** |
| 3 | memory modules… **(v6)** |
| 4 | memory modules… **(v9)** |
| 5 | memory modules… **(v3)** |

Four of the five are the **same paragraph** in different versions. Even MMR can't help much here: among the 20 closest candidates most are exact copies, so after the first two picks there is nothing different left to choose. This is a property of the dataset, not of query expansion, but it limits what any query can retrieve (see Section 6).

---

### Cell 7: Load the API key and the LLM

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("groq_api")

llm = init_chat_model("groq:openai/gpt-oss-20b")
llm
```

- `load_dotenv()` reads the `.env` file and puts its `KEY=value` lines into the environment.
- The next line copies the key from the `.env` name (`groq_api`) to the name `ChatGroq` looks for (`GROQ_API_KEY`). If `groq_api` is missing, `os.getenv` returns `None` and this line raises a `TypeError`.
- `init_chat_model("groq:openai/gpt-oss-20b")`: the part before `:` picks the provider (`groq` → `ChatGroq`), the rest is the model name, OpenAI's open-weight 20B reasoning model hosted on Groq.

The same `llm` is used **twice** in this notebook: once to expand the query, once to write the answer.

---

### Cell 8: The query expansion chain

```python
query_expansion_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Expand the following query to improve document retrieval by adding relevant synonyms, technical terms, and useful context.
Return only the expanded query, no explanation.

Original query: "{query}"

Expanded query:
""")
query_expansion_chain=query_expansion_prompt|llm|StrOutputParser()
```

| Part | Why it is there |
|---|---|
| *"adding relevant synonyms, technical terms, and useful context"* | Tells the LLM *how* to expand. This line decides the style of the expanded query. |
| *"Return only the expanded query, no explanation."* | The output goes straight into the retriever, so any extra text ("Sure! Here is…") would be searched too. |
| `{query}` | Filled with the user's question. |
| `prompt \| llm \| StrOutputParser()` | LCEL (LangChain Expression Language): the filled prompt goes to the LLM, and the parser turns the reply message into a plain string. |

---

### Cell 9: Try the expansion on its own

```python
query_expansion_chain.invoke({"query":"Langchain memory"})
```

Output:

```
'"LangChain memory" OR "LangChain memory component" OR "LangChain memory module"'
```

The model chose a **search-engine style** with `OR`. A vector retriever doesn't understand `OR`; it just embeds the whole string as text. What helps is the extra words (`component`, `module`), not the operator.

---

### Cell 10: The answer chain

```python
answer_prompt = PromptTemplate.from_template("""
Answer the question based on the context below.

Context:
{context}

Question: {input}
""")
document_chain=create_stuff_documents_chain(llm=llm,prompt=answer_prompt)
```

| Part | Why it is there |
|---|---|
| `{context}` | Filled by `create_stuff_documents_chain` with the retrieved chunks, joined by blank lines. `context` is the name that function expects by default. |
| `{input}` | Filled with the user's **original** question. |

`create_stuff_documents_chain` returns a plain string (the answer), not a dict.

---

### Cell 11: The full RAG pipeline

```python
rag_pipeline = (
    RunnableMap({
        "input":   lambda x: x["input"],
        "context": lambda x: retriever.invoke(
            query_expansion_chain.invoke({"query": x["input"]})
        ),
    })
    | document_chain
)
```

This is where query expansion is wired in. `RunnableMap` runs both entries on the same input dict:

```
{"input": "What types of memory does LangChain support?"}
      │
      ▼
RunnableMap
 ├─ "input"   → the original question, unchanged
 └─ "context" → query_expansion_chain(question) → expanded query → retriever → 5 Documents
      │
      ▼
{"input": "What types of memory ...?", "context": [Document × 5]}
      ▼
document_chain → prompt(context, input) → llm → answer string
```

Unlike [`create_retrieval_chain`](../MMR/readme.md) in the MMR notebook, this pipeline is built by hand. The reason is that the retriever must get a **different** text (the expanded query) from the one the answer prompt gets (the original question). The downside: the output is only the answer string, so the retrieved chunks and the expanded query used are not returned.

---

### Cell 12: Question 1, "What types of memory does LangChain support?"

```python
query = {"input": "What types of memory does LangChain support?"}

print("🔍 Expanded query:\n", query_expansion_chain.invoke({"query": query["input"]}))
response = rag_pipeline.invoke(query)
print("\n✅ Answer:\n", response)
```

Expanded query:

```
LangChain memory types OR memory implementations OR memory backends OR memory modules OR memory adapters
OR conversation memory OR retrieval memory OR persistent memory OR in-memory cache OR vector store
OR LangChain memory components OR LangChain memory classes OR LangChain memory support OR LangChain memory modules
```

Answer: a table with **ConversationBufferMemory** and **ConversationSummaryMemory**, and a line on choosing between full dialogue and a summary. This is correct and matches the dataset.

> **Note:** the first line calls `query_expansion_chain` to **print** an expansion, and `rag_pipeline.invoke` calls it **again** inside the pipeline. LLM output varies between calls, so the printed expanded query is not necessarily the exact one the retriever used.

---

### Cell 13: Question 2, "CrewAI agents?"

```python
query = {"input": "CrewAI agents?"}
...
```

Expanded query (shortened):

```
CrewAI agents, CrewAI bots, CrewAI assistants, AI crew management, crew scheduling AI, crew rostering AI,
airline crew planning AI, maritime crew AI, crew fatigue management AI, ... CrewAI for airlines,
CrewAI for maritime, CrewAI for aviation, CrewAI for crew safety
```

The LLM didn't know CrewAI is a multi-agent framework and guessed it was software for **airline and ship crew scheduling**. Most of the 36 added terms are about that wrong meaning.

Answer: CrewAI agents are role-based workers (researcher, writer, editor…) that collaborate, and can scale horizontally (more agents) and vertically (more reasoning depth). These points match the CrewAI paragraphs. Point 4 ("leverage fetched knowledge… reducing hallucination") is not about CrewAI at all; it comes from the RAG paragraph about LangChain and vector databases.

---

## 5. With vs Without Expansion on This Data

To see what expansion changes, the same retriever (MMR, `k=5`) was run once with the **original** question and once with the **expanded query printed above**:

#### Question 1: "What types of memory does LangChain support?"

| # | Original question | Expanded query |
|---|---|---|
| 1 | memory modules | memory modules |
| 2 | LLM providers (OpenAI, Anthropic, Cohere…) | **modular workflows: retrievers, memories, agents** |
| 3 | chains (Stuff, Map-Reduce, Refine) | memory modules |
| 4 | memory modules | memory modules |
| 5 | memory modules | memory modules |

- With the original question, two of five slots go to **off-topic** chunks (LLM providers, chains). The word "LangChain" pulls in any LangChain paragraph.
- With the expanded query, the repeated word "memory" makes every chunk memory-related. The answer has the same facts either way, because the memory paragraph is found in both cases.

#### Question 2: "CrewAI agents?"

| # | Original question | Expanded query |
|---|---|---|
| 1 | agents are defined with a purpose, goal and tools | autonomous agents with roles in structured workflows |
| 2 | autonomous agents with roles in structured workflows | **use cases: market research, legal analysis, coding** |
| 3 | purpose, goal and tools | **define a crew with YAML: roles, goals, memory, tools** |
| 4 | purpose, goal and tools | autonomous agents with roles |
| 5 | purpose, goal and tools | autonomous agents with roles |

- The original two-word question already finds the two most direct paragraphs, then fills the rest with copies.
- The expanded query brings in **two new paragraphs** (use cases, YAML configuration), even though most of its terms were about airline crews. The terms "workflow automation" and "AI agents" were enough to pull in more CrewAI content, because the dataset has nothing about airlines for the wrong terms to match.
- On a real knowledge base that *does* contain aviation or HR documents, those wrong terms would retrieve them. This is the main risk of query expansion.

---

## 6. Limitations and Tips

- **The LLM can expand in the wrong direction.** If it doesn't know a term (CrewAI here), it guesses. Give it context in the prompt, e.g. *"The documents are about LLM frameworks such as LangChain and CrewAI."*, so the expansion stays in the right domain.
- **Long expansions dilute the query.** The CrewAI expansion has 36 terms; the original two words become a small part of the embedding. Ask for a limit: *"Add at most 5–8 related terms."*
- **`OR` means nothing to a vector retriever.** The model writes Boolean search syntax, but FAISS embeds it as plain text. Ask for a natural sentence or a comma-separated list instead, or use the `OR` output with a keyword retriever (BM25) that can make use of it.
- **Keep the original question in the expanded query.** If the expansion drops or changes the user's words, retrieval can lose the main topic. A simple fix is to search with `original + " " + expanded`.
- **Every question costs an extra LLM call.** Expansion adds latency before retrieval starts. Use a small, fast model for this step, or skip expansion when the question is already long.
- **The printed expansion is not the one used.** Cells 12 and 13 call the expansion chain twice. Return it from the pipeline instead, e.g. compute the expanded query once in a `RunnableMap` step and pass it to the retriever, so it can be printed and checked.
- **The dataset is 10× duplicated.** With every paragraph repeated 10 times, most of the 5 retrieved slots are copies no matter how good the query is. Removing duplicates before indexing (or raising `fetch_k` so MMR has non-copies to choose from) shows the effect of expansion more clearly.
- **The answer prompt doesn't restrict the LLM.** "Answer the question based on the context below" is loose. Adding *"Use only the context. If the answer is not in it, say you don't know."* reduces made-up content.
- **Combine with other techniques.** Expansion raises recall (more relevant chunks found); a [reranker](../Reranking/readme.md) afterwards can push the most relevant ones to the top and cut the noise that expansion brings in.

---

## 7. Setup

From the project root:

```bash
pip install -r requirements.txt
```

The packages this folder needs are `langchain`, `langchain-classic`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-groq`, `sentence-transformers`, `faiss-cpu` and `python-dotenv`.

Create a `.env` file in the project root with your Groq key (free at [console.groq.com](https://console.groq.com)):

```
groq_api=your_groq_key_here
```

Then open `code.ipynb` **from inside the `Query-Expansion/` folder** (so `langchain_crewai_dataset.txt` is found) and run the cells in order. Cells 1–6 need no API key; Cells 9, 12 and 13 call Groq. The first run downloads the embedding model (about 90 MB).

---

## 8. Quick Glossary

| Term | Meaning |
|---|---|
| **Query expansion** | Using an LLM to add synonyms, technical terms and context to a query before retrieval |
| **Query enhancement** | The family of techniques that improve the question before retrieval (expansion, rewriting, multi-query, decomposition, HyDE) |
| **Vocabulary mismatch** | When the question and the documents describe the same thing with different words |
| **Recall** | How many of the relevant chunks the retriever actually finds |
| **Query drift** | When an expanded query moves away from what the user meant (e.g. CrewAI → airline crews) |
| **MMR (Maximal Marginal Relevance)** | Retrieval that balances relevance to the query against similarity to chunks already picked |
| **LCEL** | LangChain Expression Language: joining steps with `\|`, e.g. `prompt \| llm \| parser` |
| **`StrOutputParser`** | Turns an LLM message into a plain string |
| **`RunnableMap`** | Runs several functions on the same input and returns a dict of their results |
| **Stuff chain** | A chain that puts all retrieved chunks into one prompt |
| **`init_chat_model`** | LangChain helper that creates a chat model from a `"provider:model"` string |
| **FAISS** | Facebook AI Similarity Search, a fast in-memory vector index |
| **Groq** | A cloud provider that serves open-weight LLMs very fast, with a free tier |
