# HyDE for RAG: Searching with a Hypothetical Answer Instead of the Question

This folder covers **HyDE (Hypothetical Document Embeddings)**, a *query enhancement* technique: before retrieval, an LLM writes a **made-up answer** to the question, and that answer, not the question, is embedded and used to search the vector store.

The idea is that a question and its answer don't look alike. *"When was Steve Jobs fired from Apple?"* is a short question; the chunk that answers it is a sentence in a biography (*"In 1985, Jobs departed Apple after a long power struggle with the company's board..."*). A fake answer paragraph is written in the same style and uses the same words as the real answer, so its vector lands closer to the right chunks, **even if some of its facts are wrong**.

[Query-Expansion](../Query-Expansion/readme.md) makes the question longer with related terms, and [Query-Decomposition](../Query-Decomposition/readme.md) splits it into several sub-questions. HyDE replaces the question with an answer-shaped passage. All three change the *query*, not the chunks (compare [MMR](../MMR/readme.md) and [Reranking](../Reranking/readme.md)).

The notebook has two parts:

**Part 1: HyDE by hand (Cells 1–8)**
1. **Load Wikipedia pages**: `WikipediaLoader` fetches 5 pages about Steve Jobs, split into 84 chunks.
2. **Chroma vector store** with `all-MiniLM-L6-v2` embeddings and a plain similarity retriever (`k=5`).
3. **`get_hyde_doc()`**: asks the LLM to write a hypothetical answer.
4. **Compare**: retrieval with the raw question vs retrieval with the hypothetical answer.

**Part 2: LangChain's built-in `HypotheticalDocumentEmbedder` (Cells 9–18)**
1. **Load and chunk** `langchain_crewai_dataset.txt` into 241 chunks.
2. **Wrap the embeddings** in a `HypotheticalDocumentEmbedder`, so every *query* is turned into a hypothetical passage automatically.
3. **Full RAG pipeline**: HyDE retrieval + a stuff chain that answers from the retrieved chunks.
4. **Custom prompt**: the same embedder with your own hypothetical-answer prompt.

```
HyDE/
├── code.ipynb                    # Notebook: manual HyDE on Wikipedia + LangChain's HypotheticalDocumentEmbedder
├── langchain_crewai_dataset.txt  # 200 lines: 20 paragraphs about LangChain and CrewAI, each repeated 10 times (v1–v10)
├── output/                       # Chroma databases written by the notebook
│   ├── steve_jobs_for_hyde.db/   #   Part 1 (Wikipedia chunks)
│   └── langchain/                #   Part 2 (LangChain/CrewAI chunks)
└── readme.md                     # This file
```

---

## 1. Core Concepts

### The problem: questions and answers look different
Vector search compares the question's vector with every chunk's vector. But the embedding model was trained to place **similar texts** close together, and a question is not very similar to its answer:

| | Question | Chunk that answers it |
|---|---|---|
| Length | 7 words | 50–60 words |
| Form | *"When was ...?"* | A statement: *"In 1985, Jobs departed Apple..."* |
| Words | "fired" | "departed", "power struggle", "board", "forced out" |

So the closest chunks are often ones that share the **question's words** (here: "Steve Jobs", "Apple"), which describes almost every chunk in a Steve Jobs biography.

### HyDE
**Definition:** An extra step before retrieval in which an LLM generates a hypothetical document that answers the question. The hypothetical document is embedded, and its vector is used to search for real documents. The hypothetical document is **only used for searching**; it is never shown to the user or used as context for the answer.

```
"When was Steve Jobs fired from Apple?"
                 │  LLM: "write an answer to this"
                 ▼
 "...By 1985 ... a power struggle had erupted between Jobs and ... John Sculley ...
  The board voted to restructure the Macintosh division ... Jobs tendered his resignation ..."
                 │  embed (all-MiniLM-L6-v2)
                 ▼
          vector of the fake answer ──► similarity search ──► real chunks
                                                              │
                                     question + real chunks ──► LLM ──► final answer
```

### Why a wrong answer still works
The LLM's passage in Cell 7 contains **mistakes**: it calls Sculley Apple's "founder-CEO", gives an exact board-meeting date and a resignation date of August 24, 1985 (Jobs actually resigned in September 1985). That is fine for HyDE:

- The embedding captures **topic and style** (1985, board, Sculley, Macintosh, resignation), not whether each fact is true.
- The final answer is generated from the **real retrieved chunks**, not from the fake passage.

The paper that introduced HyDE (Gao et al., 2022, *Precise Zero-Shot Dense Retrieval without Relevance Labels*) describes it the same way: the embedding acts as a "bottleneck" that keeps the relevant pattern and drops the invented details.

### Where it sits in the pipeline

```
question ──► LLM (write hypothetical answer) ──► embed ──► vector store search ──► top-k chunks
    │                                                                                 │
    └──────────────────────────────────────────────────────────────► answer LLM ◄─────┘
```

Only the **query side** changes. The chunks are embedded normally, once, when the vector store is built.

### Query enhancement family

| Technique | What the LLM produces | Searches per question |
|---|---|---|
| Query expansion | One longer query with synonyms and related terms | 1 |
| Multi-query | Several rephrasings of the same question; results are merged | several |
| Query decomposition | Several different sub-questions, one per topic | one per sub-question |
| **HyDE** (this folder) | A fake **answer** paragraph, embedded instead of the question | 1 |

Expansion and decomposition still search with *questions*. HyDE is the only one that searches with an *answer*.

### HyDE vs searching with the question

| | Raw question | HyDE |
|---|---|---|
| LLM calls per question | 1 (to answer) | 2 (hypothetical doc + answer) |
| Short or vague questions | Few words to match on | The passage adds the vocabulary of a real answer |
| Questions phrased differently from the documents | Wording mismatch hurts | The passage is written like a document |
| Topics the LLM doesn't know | Unaffected | The passage can drift to the wrong topic and retrieve the wrong chunks |
| Exact names, codes, IDs | Matched directly | May be paraphrased or replaced in the passage |
| Latency | One embedding | One LLM generation (often several seconds) + one embedding |

---

## 2. Use Cases

HyDE helps when the **question doesn't look like the documents**:

- **Short or keyword-style questions**: *"Jobs fired Apple when?"* The passage turns it into a full paragraph.
- **Vocabulary mismatch**: users say "fired", documents say "forced out" or "departed after a power struggle". The LLM knows both phrasings.
- **Zero-shot retrieval**: you have no labelled question–answer pairs to fine-tune the embedding model. HyDE improves retrieval without any training.
- **Domain-specific corpora** (scientific papers, legal text, financial reports) where documents are written in a formal style very different from user questions. LangChain's built-in prompts (`sci_fact`, `fiqa`, `trec_covid`...) are made for these.
- **Questions in a different language** from the documents: the LLM can write the hypothetical passage in the documents' language.

It is **less useful** when:
- The topic is private or very new (your company's internal tools, last week's news). The LLM can't write a plausible answer, so the passage may describe something else entirely.
- The question depends on exact strings: error codes, product IDs, names. Use keyword search or a [hybrid retriever](../Hybrid-Retriever/readme.md) instead.
- Latency matters. HyDE adds a full LLM generation before every search.
- The question already looks like the documents. Then the extra call gains little, as in Part 2 below.

---

## 3. Libraries Used

| Library | Purpose in this notebook |
|---|---|
| `langchain` | `init_chat_model`, which creates a chat model from a `"provider:model"` string |
| `langchain-groq` | `ChatGroq`, created behind the scenes by `init_chat_model("groq:...")` |
| `langchain-core` | `ChatPromptTemplate`, `SystemMessagePromptTemplate` and `PromptTemplate` |
| `langchain-community` | `WikipediaLoader`, `TextLoader` and the `Chroma` vector store wrapper |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` to cut the text into chunks |
| `langchain-huggingface` | `HuggingFaceEmbeddings`, the LangChain wrapper around sentence-transformers |
| `sentence-transformers` | Downloads and runs the `all-MiniLM-L6-v2` embedding model locally |
| `chromadb` | The vector database behind `Chroma`; saves the index to disk under `output/` |
| `wikipedia` | The Python client `WikipediaLoader` uses to search and download pages |
| `langchain-classic` | `HypotheticalDocumentEmbedder` and `create_stuff_documents_chain` (moved here from `langchain` in LangChain 1.x) |
| `python-dotenv` | Loads the Groq API key from a `.env` file |

---

## 4. The Code, Cell by Cell

### Cell 0: What is HyDE? (Markdown)
A short definition: generate a hypothetical answer with an LLM, embed it, and search with that embedding instead of the query's. It lists when this helps: short queries, a language mismatch between query and documents, and retrieving by *answer content* rather than question words.

---

## Part 1: HyDE by Hand

### Cell 1: Imports

```python
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
```

| Import | What it is |
|---|---|
| `WikipediaLoader` | Searches Wikipedia and loads the top pages as `Document`s (page text + title, summary and URL in the metadata). |
| `RecursiveCharacterTextSplitter` | Splits text into chunks of a maximum size, preferring to break at paragraphs, then lines, then spaces. |
| `HuggingFaceEmbeddings` | Turns text into vectors with a local sentence-transformers model. |
| `Chroma` | LangChain's wrapper around the Chroma vector database. Unlike FAISS in earlier folders, it can save the index to a folder on disk. |

The `DeprecationWarning` that `langchain-community` is being sunset is only a notice; everything still works.

---

### Cell 2: Load and chunk Wikipedia pages

```python
import wikipedia
from datetime import timedelta

wikipedia.set_user_agent("RAG-learning-notebook/1.0")
wikipedia.set_rate_limiting(True, min_wait=timedelta(seconds=1))

chunk_size = 300
chunk_overlap = 100

loader = WikipediaLoader(query="Steve Jobs", load_max_docs=5)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
docs = text_splitter.split_documents(documents=documents)
print(len(docs))
docs[:3]
```

| Line | What it does |
|---|---|
| `set_user_agent(...)` | Wikipedia rejects the `wikipedia` package's default User-Agent with **HTTP 429 (Too Many Requests)**. Any descriptive name of your own fixes it. |
| `set_rate_limiting(True, min_wait=1s)` | Waits at least one second between requests, so loading 5 pages doesn't get blocked. |
| `WikipediaLoader(query="Steve Jobs", load_max_docs=5)` | Searches Wikipedia for "Steve Jobs" and loads the top 5 results (the main biography plus related pages, such as the 2015 film *Steve Jobs*). |
| `chunk_size=300`, `chunk_overlap=100` | Chunks of at most 300 characters; when a paragraph is cut, up to 100 characters repeat in the next chunk. |

Output: **84 chunks**. Each `Document`'s metadata carries the page `title`, its `summary` and `source` URL.

---

### Cell 3: Embedding model

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

The small, free embedding model used throughout this repo (384-number vectors, runs on CPU). The first run downloads about 90 MB. The `Loading weights` progress bar is normal.

---

### Cell 4: Load the API key and the LLM

```python
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model

os.environ["GROQ_API_KEY"] = os.getenv("groq_api")
llm = init_chat_model("groq:openai/gpt-oss-20b")
```

- `load_dotenv()` reads the `.env` file and puts its `KEY=value` lines into the environment.
- The next line copies the key from the `.env` name (`groq_api`) to the name `ChatGroq` looks for (`GROQ_API_KEY`). If `groq_api` is missing, `os.getenv` returns `None` and this line raises a `TypeError`.
- `init_chat_model("groq:openai/gpt-oss-20b")`: `groq` picks the provider, the rest is the model name (OpenAI's open-weight 20B model hosted on Groq).

The same `llm` writes the hypothetical documents **and** answers the questions.

---

### Cell 5: Vector store and retriever

```python
db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="output/steve_jobs_for_hyde.db")
base_retriever = db.as_retriever(search_kwargs={"k": 5})
```

| Line | What it does |
|---|---|
| `Chroma.from_documents(...)` | Embeds all 84 chunks with the **normal** embedding model and stores them. |
| `persist_directory=...` | Saves the database to `output/steve_jobs_for_hyde.db/` so it survives a kernel restart. |
| `as_retriever(search_kwargs={"k": 5})` | Plain similarity search returning the 5 closest chunks. No MMR, so the comparison in Cell 8 shows the effect of HyDE alone. |

**Re-running this cell adds the 84 chunks again** to the same saved collection, so results start showing duplicates. Delete the `output/` folder before re-running, or load the existing store with `Chroma(persist_directory=..., embedding_function=embeddings)`.

---

### Cell 6: `get_hyde_doc()`

```python
def get_hyde_doc(query):
    template = """Imagine you are an expert writing a detailed explanation on the topic: '{query}'
    create a hypothetical answer for the topic"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(template=template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
    messages = chat_prompt.format_prompt(query=query).to_messages()
    response = llm.invoke(messages)
    hypo_doc = response.content
    return hypo_doc
```

| Step | What it does |
|---|---|
| `template` | Asks the LLM to play an expert and write a **detailed** answer. Detailed means many topic words for the embedding to pick up. |
| `SystemMessagePromptTemplate` / `ChatPromptTemplate` | Wraps the template as a system message in a chat prompt. |
| `format_prompt(query=query).to_messages()` | Fills in `{query}` and returns the list of messages to send. |
| `llm.invoke(messages).content` | Calls the LLM and keeps only the text. |

`SystemMessagePromptTemplate` and `ChatPromptTemplate` are imported in Cell 7, **after** this function is defined. That works because Python only looks the names up when the function is *called*, and Cell 7 imports them before calling it. Moving the imports into this cell would be cleaner.

---

### Cell 7: Generate the hypothetical document

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate
query = "When was Steve Jobs fired from Apple?"
hypo_doc = get_hyde_doc(query=query)
print(hypo_doc)
```

The output is a long Markdown article: a timeline table (1976–1985), a section on *"The Board Meeting – March 1985"*, one on *"The Decision to Leave – August 1985"* and one on *"Why It Was 'Fired' and Not Just 'Resigned'"*.

It is the right **topic** and uses the right **vocabulary** (1985, board, John Sculley, Macintosh division, power struggle, resignation), but several **facts are invented or wrong**: Sculley is called Apple's "founder-CEO", the meeting gets an exact date, and the resignation is dated August 24, 1985. As explained in [Section 1](#why-a-wrong-answer-still-works), this doesn't matter: the passage is only used to search.

`all-MiniLM-L6-v2` reads only the **first 256 tokens** (roughly 1,000 characters) of any text and ignores the rest. So only the title, the timeline table and the start of the board-meeting section actually reach the vector. For HyDE, a short passage is usually as good as a long one, and faster to generate.

---

### Cell 8: Raw question vs hypothetical document

```python
print("🔎 Normal retrieval (raw query):")
for d in base_retriever.invoke(query):
    print("-", d.page_content[:120])

print("\n🧪 HyDE retrieval (hypothetical doc):")
matched_doc = base_retriever.invoke(hypo_doc)
for d in matched_doc:
    print("-", d.page_content[:120])
```

The retriever is the same; only the text passed to `invoke` changes. Output:

| # | Raw question | HyDE (hypothetical doc) |
|---|---|---|
| 1 | *"1986, becoming its chairman ... Jobs returned to Apple in 1997 as CEO..."* | ✅ *"In 1985, Jobs departed Apple after a long power struggle with the company's board and its then-CEO, John Sculley..."* |
| 2 | ✅ *"In 1985, Jobs departed Apple after a long power struggle..."* | *"Steven Paul Jobs (February 24, 1955 – October 5, 2011) was an American businessman..."* |
| 3 | *"Steven Paul Jobs (February 24, 1955 – October 5, 2011)..."* | *"1986, becoming its chairman ... Jobs returned to Apple in 1997..."* |
| 4 | *"Apple CEO John Sculley demands to know why the world believes he fired Jobs – Jobs was actually forced out by the Apple board..."* | *"Apple CEO John Sculley demands to know why the world believes he fired Jobs..."* |
| 5 | *"In 1997, Jobs returned to Apple as CEO after the company acquired NeXT..."* | *"conducted by Sorkin. The film covers fourteen years in the life of Apple Inc. co-founder Steve Jobs..."* |

What this shows:
- **The answer chunk moved to rank 1.** With the raw question, the top chunk is about Jobs *returning* to Apple in 1997, because "Jobs", "Apple" and "CEO" match the question's words. HyDE's passage is about 1985, the board and Sculley, so the 1985 chunk wins.
- **The 1997 "returned to Apple" chunk dropped out.** It shares the question's words but not the answer's topic.
- **The gain is modest here.** Both lists contain the answer chunk, because "Steve Jobs" + "Apple" + "fired" is already a good query on a corpus that is all about Steve Jobs. HyDE matters more when the right chunk is outside the top-k for the raw question.
- **A slightly odd 5th result.** The Sorkin film chunk appears because the passage is long and general about Jobs' career. A shorter, more focused hypothetical answer would avoid this.

---

## Part 2: LangChain's `HypotheticalDocumentEmbedder`

### Cell 9: Section title (Markdown)

LangChain has HyDE built in as an **embeddings wrapper**: you give it an LLM and your normal embeddings, and it behaves like an embeddings object whose `embed_query` writes and embeds a hypothetical document.

---

### Cell 10: Imports and load the text file

```python
from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

loader = TextLoader("langchain_crewai_dataset.txt", encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(len(chunks))
```

| Line | What it does |
|---|---|
| `HypotheticalDocumentEmbedder` | The HyDE embeddings wrapper. In LangChain 1.x it lives in `langchain-classic`. |
| `create_stuff_documents_chain` | Builds a chain that "stuffs" a list of documents into one prompt variable and calls the LLM. |
| `encoding="utf-8"` | The file contains a curly apostrophe (`LangChain’s`). Without it, Windows may fail to read the file. The path is relative, so run the notebook from inside `HyDE/`. |
| `chunk_size=300`, `chunk_overlap=50` | Same settings as the Query-Expansion and Query-Decomposition folders. |

Output: **241 chunks**.

#### About the dataset
The file is the same one used in [Query-Expansion](../Query-Expansion/readme.md) and [Query-Decomposition](../Query-Decomposition/readme.md): **200 lines**, but only **20 different paragraphs** (10 about LangChain, 10 about CrewAI), each repeated **10 times** and tagged `(v1)` to `(v10)`. This is why every search below returns 4 copies of the same paragraph.

---

### Cell 11: Base embeddings

```python
base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

The same model as Cell 3 (loaded a second time; `embeddings` from Cell 3 could be reused). The `unauthenticated requests to the HF Hub` warning is normal; set an `HF_TOKEN` only if you hit download rate limits.

---

### Cell 12: Built-in prompts (Markdown)

`HypotheticalDocumentEmbedder` ships with one prompt per dataset from the HyDE paper's benchmarks, chosen with `prompt_key`:

| `prompt_key` | Asks the LLM to write... |
|---|---|
| `web_search` | a passage that answers the question (general purpose) |
| `sci_fact` | a scientific paper passage that supports or refutes a claim |
| `arguana` | a counter-argument to a passage |
| `trec_covid` | a scientific paper passage about COVID-19 |
| `fiqa` | a financial article passage |
| `dbpedia_entity` | a passage that answers the question (almost the same as `web_search`) |
| `trec_news` | a news passage about a topic |
| `mr_tydi` | a detailed passage in Swahili/Korean/Japanese/Bengali (multilingual benchmark) |

The `web_search` prompt is simply:

```
Please write a passage to answer the question
Question: {QUESTION}
Passage:
```

---

### Cell 13: Create the HyDE embedder

```python
hyde_embedding_function = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    prompt_key="web_search"
)
```

The resulting object has the same two methods as any LangChain embeddings, but they behave differently:

| Method | Called when | What it does |
|---|---|---|
| `embed_documents(texts)` | Building the vector store | **Passes straight through** to `base_embeddings`. Chunks are embedded normally; no LLM calls. |
| `embed_query(text)` | Every search | Runs `prompt \| llm` to write a hypothetical passage, embeds it with `base_embeddings`, and returns that vector. |

If the chain produces several passages, `embed_query` averages their vectors (`combine_embeddings`). With one LLM call, as here, the average is just the one vector.

---

### Cell 14: Vector store

```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=hyde_embedding_function,
    persist_directory="output/langchain"
)
```

Because `embed_documents` passes through, **indexing 241 chunks costs no LLM calls**; the chunk vectors are identical to what `base_embeddings` would give. The HyDE behaviour only kicks in at search time, since Chroma calls `embed_query` on every `similarity_search`.

As in Cell 5, re-running this cell appends another 241 chunks to `output/langchain/`.

---

### Cell 15: The answer chain

```python
rag_prompt = PromptTemplate.from_template("""
Use the context below to answer the question.

Context:
{context}

Question: {input}
""")
rag_chain = create_stuff_documents_chain(llm=llm, prompt=rag_prompt)
```

`create_stuff_documents_chain` joins the `page_content` of every document passed as `context` into the `{context}` slot, fills `{input}` with the question, calls the LLM and returns a string. Note that the answer is generated from the **original question** and the **real chunks**; the hypothetical passage never appears here.

---

### Cell 16: The HyDE RAG pipeline

```python
def hyde_rag_pipeline(query):
    matched_docs = vectorstore.similarity_search(query, k=4)   # query → LLM passage → embedding → search
    for d in matched_docs:
        print("-", d.page_content[:100])
    response = rag_chain.invoke({
        "input": query,
        "context": matched_docs
    })
    return response
```

| Step | LLM call? | What happens |
|---|---|---|
| `vectorstore.similarity_search(query, k=4)` | ✅ | Chroma calls `hyde_embedding_function.embed_query(query)`, which writes a hypothetical passage and embeds it; the 4 closest chunks are returned. |
| `print(...)` | | Shows the first 100 characters of each retrieved chunk. |
| `rag_chain.invoke(...)` | ✅ | Answers the original question from the retrieved chunks. |

Two LLM calls per question. The hypothetical passage is created inside `embed_query` and thrown away, so it isn't printed. To see it, call `hyde_embedding_function.llm_chain.invoke({"QUESTION": query})`.

---

### Cell 17: Run it

```python
query = "What memory modules does LangChain provide?"
answer = hyde_rag_pipeline(query)
print("\n✅ Final Answer:\n", answer)
```

Output:

```
- LangChain offers memory modules like ConversationBufferMemory and ConversationSummaryMemory. These a
- LangChain offers memory modules like ConversationBufferMemory and ConversationSummaryMemory. These a
- LangChain offers memory modules like ConversationBufferMemory and ConversationSummaryMemory. These a
- LangChain offers memory modules like ConversationBufferMemory and ConversationSummaryMemory. These a

✅ Final Answer:
 LangChain provides two primary memory modules:

1. **ConversationBufferMemory** – Keeps a running buffer of the entire conversation history, ...
2. **ConversationSummaryMemory** – Stores a concise summary of the conversation, ...
```

- **The right paragraph is found**, four times over (the dataset's duplicates, and no MMR to filter them).
- **The answer is grounded.** It names exactly the two modules in the document, even though the hypothetical passage almost certainly listed others (`ConversationBufferWindowMemory`, `VectorStoreRetrieverMemory`, ...). Invented details in the passage don't leak into the answer.
- This question already uses the paragraph's own words ("memory modules", "LangChain"), so the raw question would retrieve the same chunks. It shows the pipeline works, not that HyDE was needed.

---

### Cell 18: A custom HyDE prompt

```python
custom = PromptTemplate.from_template(
    "Generate a concise hypothetical answer for this topic: {query}"
)

hyde_embedding_function = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    custom_prompt=custom
)

qvec = hyde_embedding_function.embed_query(query)
for d in vectorstore.similarity_search_by_vector(qvec, k=4):
    print("-", d.page_content[:100])
```

| Line | What it does |
|---|---|
| `custom_prompt=custom` | Replaces the built-in `web_search` prompt. The variable can have any name (`{query}` here, `{QUESTION}` in the built-ins); the embedder fills whichever single variable the prompt has. |
| *"concise"* | A short passage is faster to generate and stays within the embedding model's 256-token limit. |
| `embed_query(query)` | Writes and embeds the hypothetical answer with the new prompt. |
| `similarity_search_by_vector(qvec, k=4)` | Searches with a ready-made vector. The store from Cell 14 is reused: the chunk vectors came from `base_embeddings` and don't depend on the prompt, so there's no need to rebuild it. |

The store still holds the **old** embedder from Cell 14, so `vectorstore.similarity_search(query)` would still use the `web_search` prompt. That is why this cell embeds the query itself and searches by vector.

Output: the same 4 copies of the memory-modules paragraph as Cell 17.

---

## 5. With vs Without HyDE on This Data

| | Raw question | HyDE |
|---|---|---|
| **Steve Jobs** (*"When was Steve Jobs fired from Apple?"*, `k=5`) | Answer chunk at rank 2; rank 1 is about the 1997 return | ✅ Answer chunk at rank 1; the 1997 chunk drops out, a film chunk comes in |
| **LangChain** (*"What memory modules does LangChain provide?"*, `k=4`) | Memory paragraph (the question already matches its words) | Memory paragraph ×4 (no change) |
| LLM calls | 1 (answer) | 2 (hypothetical doc + answer) |

HyDE **re-ranked the Wikipedia results towards the actual answer**, at the cost of one extra LLM call. On the LangChain dataset it made no difference, because the question and the paragraph already share their key words. HyDE is worth its cost when the question and the documents are phrased differently.

---

## 6. Limitations and Tips

- **The LLM must know something about the topic.** HyDE works because the LLM can write a plausible answer. On private or brand-new topics the passage may be about something else, and retrieval gets *worse*. Compare HyDE and raw-question retrieval on your own data before switching.
- **Keep the passage short.** `all-MiniLM-L6-v2` only reads the first 256 tokens. The Cell 6 prompt asks for a *detailed* explanation and gets a long article with tables, most of which is ignored. *"Write a short paragraph that answers..."* (like Cell 18) is faster and just as good for retrieval.
- **Match the passage to the documents' style.** For Wikipedia, *"Write an encyclopedia paragraph that answers..."*; for code docs, *"Write a documentation snippet..."*. The closer the fake passage looks to a real chunk, the better.
- **Never show the hypothetical document to the user.** It can contain confident mistakes (Cell 7). Only answer from the retrieved chunks, as Cells 15–17 do.
- **Every search costs an LLM call.** Cache hypothetical passages for repeated questions, use a small fast model for them, or only use HyDE when a normal search scores poorly.
- **Exact terms can get lost.** If the question contains an ID, error code or rare name, the LLM may paraphrase it away. Combine HyDE with keyword search ([Hybrid-Retriever](../Hybrid-Retriever/readme.md)), or search with both the question and the passage and merge the results.
- **Generate several passages for stability.** The original paper writes several hypothetical documents and averages their vectors (plus the question's own vector). One passage can be unlucky; the average is smoother. `HypotheticalDocumentEmbedder` averages automatically if its chain returns several.
- **Chroma's saved folders grow on every run.** `Chroma.from_documents(..., persist_directory=...)` appends to the existing collection. Delete `output/` before re-running, or load the existing store instead of rebuilding it.
- **The LangChain dataset is 10× duplicated.** Every search returns 4 copies of one paragraph. Use [MMR](../MMR/readme.md) (`search_type="mmr"`) or remove duplicates before indexing to get 4 *different* paragraphs.
- **Combine with other techniques.** HyDE can be applied per sub-question after [decomposition](../Query-Decomposition/readme.md), and its results can be [reranked](../Reranking/readme.md) with a cross-encoder against the *original* question.

---

## 7. Setup

From the project root:

```bash
pip install -r requirements.txt
```

The packages this folder needs are `langchain`, `langchain-classic`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-groq`, `sentence-transformers`, `chromadb`, `wikipedia` and `python-dotenv`.

Create a `.env` file in the project root with your Groq key (free at [console.groq.com](https://console.groq.com)):

```
groq_api=your_groq_key_here
```

Then open `code.ipynb` **from inside the `HyDE/` folder** (so `langchain_crewai_dataset.txt` is found and the databases land in `HyDE/output/`) and run the cells in order. Cell 2 needs internet access for Wikipedia. Cells 7, 17 and 18 call Groq. The first run downloads the embedding model (about 90 MB).

---

## 8. Quick Glossary

| Term | Meaning |
|---|---|
| **HyDE** | Hypothetical Document Embeddings: searching with the embedding of an LLM-written answer instead of the question |
| **Hypothetical document** | The fake answer passage the LLM writes; used only for searching, never as context or output |
| **`HypotheticalDocumentEmbedder`** | LangChain's HyDE wrapper: normal `embed_documents`, LLM-powered `embed_query` |
| **`prompt_key`** | Picks one of the built-in HyDE prompts (`web_search`, `sci_fact`, `fiqa`, ...) |
| **`custom_prompt`** | Replaces the built-in prompt with your own `PromptTemplate` |
| **Vocabulary mismatch** | When the question and the relevant document use different words for the same thing |
| **Zero-shot retrieval** | Retrieval without any labelled examples or fine-tuning for the target data |
| **Query enhancement** | The family of techniques that improve the question before retrieval (expansion, rewriting, multi-query, decomposition, HyDE) |
| **Chroma** | An open-source vector database that can save its index to disk (`persist_directory`) |
| **`WikipediaLoader`** | LangChain loader that searches Wikipedia and returns the top pages as `Document`s |
| **`similarity_search_by_vector`** | Searching a vector store with a vector you computed yourself instead of a text query |
| **Stuff chain** | A chain that puts all retrieved chunks into one prompt |
| **`init_chat_model`** | LangChain helper that creates a chat model from a `"provider:model"` string |
| **Groq** | A cloud provider that serves open-weight LLMs very fast, with a free tier |
