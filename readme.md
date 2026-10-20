# RAG From Scratch — Data Ingestion & Parsing with LangChain

A hands-on, notebook-based study of **Retrieval-Augmented Generation (RAG)**, built step by step with [LangChain](https://python.langchain.com/). This repository currently covers the **first and most important stage** of every RAG system: getting data out of real-world files (plain text, PDF, Word, CSV, Excel, JSON) and turning it into clean, well-labelled, correctly sized chunks ready for embedding.

Each folder is a self-contained lesson: a Jupyter notebook with runnable experiments, the sample input files it works on, and a detailed `readme.md` that explains every concept and every cell.

---

## Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [The RAG Pipeline, Stage by Stage](#2-the-rag-pipeline-stage-by-stage)
3. [What This Repository Covers](#3-what-this-repository-covers)
4. [Repository Structure](#4-repository-structure)
5. [Learning Path & Module Summaries](#5-learning-path--module-summaries)
6. [Core Concepts Used Everywhere](#6-core-concepts-used-everywhere)
7. [Choosing a Loader: Cross-Format Comparison](#7-choosing-a-loader-cross-format-comparison)
8. [Setup](#8-setup)
9. [Dependencies Explained](#9-dependencies-explained)
10. [Troubleshooting](#10-troubleshooting)
11. [Roadmap](#11-roadmap)
12. [Glossary](#12-glossary)
13. [References](#13-references)

---

## 1. What is RAG?

**Definition:** Retrieval-Augmented Generation is a technique in which a Large Language Model (LLM) answers a question using information **retrieved from your own data at query time**, instead of relying only on what it memorised during training.

**Why it exists.** An LLM on its own has three big limits:

| Problem | What it means | How RAG helps |
|---|---|---|
| **Knowledge cut-off** | The model knows nothing that happened after its training data was collected. | Fresh documents are retrieved at query time. |
| **No private knowledge** | The model has never seen your company wiki, contracts, product catalogue or codebase. | Your own documents become the knowledge source. |
| **Hallucination** | When unsure, the model may confidently invent an answer. | The answer is *grounded* in retrieved text, and you can cite the source chunk. |

**RAG vs fine-tuning.** Fine-tuning changes the model's weights, which makes it good for teaching *style, format or behaviour*. RAG leaves the model untouched and changes the *context* it sees, which makes it good for teaching *facts*. RAG is cheaper, updates instantly (just re-index a document), and is auditable because every answer can point to its sources. In practice the two are often combined.

The idea was introduced in *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020) and has since become the standard architecture for "chat with your documents" systems.

---

## 2. The RAG Pipeline, Stage by Stage

A RAG system has two phases: an **offline indexing phase** that prepares your data once, and an **online query phase** that runs for every question.

```
                         ┌──────────────── INDEXING (offline, once per document) ────────────────┐
                         │                                                                       │
 Raw files ──► Load / Parse ──► Clean ──► Chunk ──► Embed ──► Store in Vector DB                 │
 (.txt .pdf     (Document        (fix       (split    (text →    (FAISS, Chroma, …)              │
  .docx .csv     objects +        noise)     into      vectors)                                  │
  .xlsx .json)   metadata)                   pieces)                                             │
                         │◄──────────── this repository ────────────►│                           │
                         └───────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────── QUERY (online, every question) ───────────────────┐
 User question ──► Embed question ──► Similarity search in Vector DB ──► Top-k chunks             │
                                                                              │                   │
                         Prompt = instructions + retrieved chunks + question ◄┘                   │
                                                                              │                   │
                                                                   LLM ──► Grounded answer        │
                         └───────────────────────────────────────────────────────────────────────┘
```

| Stage | What happens | Key decisions |
|---|---|---|
| **1. Load / Parse** | Read a file and convert it into `Document` objects (text + metadata). | Which loader? One Document per page, row, sheet, element or record? |
| **2. Clean** | Remove extraction noise: broken line breaks, ligatures, headers/footers, extra whitespace. | How aggressive? Never remove meaning along with the noise. |
| **3. Chunk** | Split long text into pieces small enough to embed and to fit in the prompt. | `chunk_size`, `chunk_overlap`, which separators, keep structure? |
| **4. Embed** | Turn each chunk into a vector (a list of numbers) that captures its meaning. | Which embedding model? (e.g. `sentence-transformers`, OpenAI) |
| **5. Store** | Save vectors + text + metadata in a vector database. | FAISS (in-memory, fast) vs Chroma (persistent, metadata filters) |
| **6. Retrieve** | Embed the question and find the most similar chunks. | `k`, similarity threshold, metadata filters, hybrid search (BM25 + vectors), re-ranking |
| **7. Generate** | Put the retrieved chunks into the prompt and ask the LLM to answer from them. | Prompt template, which LLM (e.g. Groq, OpenAI), citations |

> **Garbage in, garbage out.** Retrieval can only find what parsing extracted correctly. A price separated from its column name, a heading lost from a Word file, or a PDF figure turned into one-word-per-line noise all damage every stage that follows. That is why this repository spends so much time on stages 1–3.

---

## 3. What This Repository Covers

| Stage | Status | Where |
|---|---|---|
| Load / Parse plain text | ✅ Done | [DataIngestion/](DataIngestion/) |
| Chunking strategies | ✅ Done | [DataIngestion/](DataIngestion/), [DataParsing/](DataParsing/) |
| Load / Parse / Clean PDF | ✅ Done | [DataParsing/](DataParsing/) |
| Load / Parse Word (`.docx`) | ✅ Done | [ParsingWordDoc/](ParsingWordDoc/) |
| Load / Parse CSV & Excel | ✅ Done | [ParsingCSV/](ParsingCSV/) |
| Load / Parse JSON & JSON Lines | ✅ Done | [JSONParsing/](JSONParsing/) |
| Embeddings | 🔜 Planned | — |
| Vector stores (FAISS, Chroma) | 🔜 Planned | — |
| Retrieval (similarity, hybrid/BM25) | 🔜 Planned | — |
| Generation with an LLM (Groq / OpenAI) | 🔜 Planned | — |
| Agentic RAG (LangGraph) | 🔜 Planned | — |

The packages for the planned stages are already listed in [requirements.txt](requirements.txt) (see [Dependencies Explained](#9-dependencies-explained)).

---

## 4. Repository Structure

```
RAG/
├── readme.md                     # This file: project overview, concepts, setup
├── requirements.txt              # All Python dependencies (current + planned stages)
├── .gitignore                    # Keeps .env (API keys) out of git
├── .env                          # Your API keys (not committed, create it yourself)
├── .venv/                        # Local virtual environment (not committed)
│
├── DataIngestion/                # Lesson 1: Documents, text loaders, text splitters
│   ├── code.ipynb
│   ├── data/text_files/
│   │   ├── python_intro.txt
│   │   └── machine_learning.txt
│   └── readme.md
│
├── DataParsing/                  # Lesson 2: PDF loading, cleaning, chunking pipeline
│   ├── code.ipynb
│   ├── pdf/attention.pdf         # "Attention Is All You Need" (15 pages)
│   └── readme.md
│
├── ParsingWordDoc/               # Lesson 3: Word (.docx) loaders
│   ├── code.ipynb
│   ├── docx/
│   │   ├── proposal.docx
│   │   └── proposal.txt
│   └── readme.md
│
├── ParsingCSV/                   # Lesson 4: CSV & Excel (tabular data)
│   ├── code.ipynb
│   ├── Files/
│   │   ├── products.csv
│   │   └── inventory.xlsx
│   └── readme.md
│
└── JSONParsing/                  # Lesson 5: JSON & JSON Lines (nested data)
    ├── code.ipynb
    ├── files/
    │   ├── company_data.json
    │   └── events.jsonl
    └── readme.md
```

---

## 5. Learning Path & Module Summaries

Work through the folders in this order. Each one builds on the concepts of the previous one.

| # | Module | Input format | Main tools | Key lesson |
|---|---|---|---|---|
| 1 | [DataIngestion](DataIngestion/readme.md) | `.txt` | `Document`, `TextLoader`, `DirectoryLoader`, `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, `TokenTextSplitter` | What a Document is and how chunk size, overlap and separators change the chunks you get |
| 2 | [DataParsing](DataParsing/readme.md) | `.pdf` | `PyPDFLoader`, `PyMuPDFLoader`, `UnstructuredPDFLoader`, custom `SmartPDFProcessor` | PDFs store positioned glyphs, not paragraphs, so extracted text needs cleaning before chunking |
| 3 | [ParsingWordDoc](ParsingWordDoc/readme.md) | `.docx` | `Docx2txtLoader`, `UnstructuredWordDocumentLoader` | Word files keep their logical structure, and an element-aware loader can use it |
| 4 | [ParsingCSV](ParsingCSV/readme.md) | `.csv`, `.xlsx` | `CSVLoader`, pandas, `UnstructuredExcelLoader` | With tables, *what counts as one Document* (row, sheet, group) is the main design decision |
| 5 | [JSONParsing](JSONParsing/readme.md) | `.json`, `.jsonl` | `JSONLoader` + `jq_schema`, custom flattening | Nested data gets its meaning from its path, so flatten it into readable, labelled text |

### 5.1 DataIngestion — Documents, Loaders and Text Splitters

The foundation for everything else.

- **`Document`** = `page_content` (the text that gets embedded) + `metadata` (a dict such as `source`, `page`, `author`).
- **`TextLoader`** loads one file. **`DirectoryLoader`** loads every file matching a glob such as `**/*.txt`.
- **Text splitters** compared side by side:
  - `CharacterTextSplitter`: splits on one separator.
  - `RecursiveCharacterTextSplitter`: tries `"\n\n"`, then `"\n"`, then `" "`, then `""`, falling back to smaller separators only when needed. **This is the default choice for most text.**
  - `TokenTextSplitter`: measures size in model tokens (via `tiktoken`) instead of characters.
- Shows how `chunk_overlap` repeats the end of one chunk at the start of the next, so context is not lost at boundaries.

➡ Full details: [DataIngestion/readme.md](DataIngestion/readme.md)

### 5.2 DataParsing — PDF Loading, Cleaning and Chunking

Uses the *Attention Is All You Need* paper as a realistic, hard-to-parse input.

- Compares **`PyPDFLoader`** (pure Python, easy), **`PyMuPDFLoader`** (C-based, fast, rich metadata) and **`UnstructuredPDFLoader`** (layout-aware, OCR, slower).
- Shows common PDF extraction problems: broken line breaks, extra whitespace, **ligatures** (`ﬁ` → `fi`), and figure text that comes out one word per line.
- Builds a `clean_text` function and a full **`SmartPDFProcessor`** class: *load → clean → skip near-empty pages → chunk → enrich metadata* (1-indexed `page`, `total_pages`, `chunk_method`, `char_count`). Result: 15 pages → **49 chunks**.
- Lists known issues and improvements (trailing-comma tuple bug, space-only separators, chunks never crossing page breaks).

➡ Full details: [DataParsing/readme.md](DataParsing/readme.md)

### 5.3 ParsingWordDoc — Word Documents

- A `.docx` is a ZIP of XML files that **keeps logical structure** (headings, lists, tables, styles), so it is usually easier and more accurate to parse than a PDF.
- **`Docx2txtLoader`** returns 1 flat Document holding all the text: fast and light.
- **`UnstructuredWordDocumentLoader(mode="elements")`** returns 20 typed elements (`Title`, `NarrativeText`, `ListItem`, `Table`, …) with parent links, depth and HTML for tables. Use it for structure-aware chunking and filtering.

➡ Full details: [ParsingWordDoc/readme.md](ParsingWordDoc/readme.md)

### 5.4 ParsingCSV — CSV and Excel Files

- A value like `29.99` means nothing without its column (`Price`) and its row (`Mouse`), so the way a table is turned into text matters a lot.
- **`CSVLoader`**: one Document per row as `column: value` lines (5 Documents).
- **Custom pandas processing**: one Document per row, with your own wording, real data types and custom metadata.
- **Custom Excel processing**: one Document per **sheet** (2 Documents: `Products`, `Summary`).
- **`UnstructuredExcelLoader`**: one table element per sheet with an HTML copy in the metadata (needs `unstructured[xlsx]`).
- Explains why **aggregation questions** ("total stock of all electronics") are answered poorly by vector search alone, and what to do instead: pre-computed summaries, or a pandas/SQL tool.

➡ Full details: [ParsingCSV/readme.md](ParsingCSV/readme.md)

### 5.5 JSONParsing — JSON and JSON Lines

- JSON is a **tree**: every value gets its meaning from its path (`employees[0].projects[1].status`).
- **`JSONLoader` + `jq_schema`**: a [jq](https://jqlang.github.io/jq/) query selects the records, and each match becomes a Document (`json_lines=True` for `.jsonl`, `text_content=False` for non-string content, `content_key` and `metadata_func` for shaping).
- **Custom processing**: plain Python walks the tree and writes each employee as a readable, labelled profile, with metadata you design yourself.
- Covers generic flattening, keeping parent context in child records, and using the output with metadata filters.

➡ Full details: [JSONParsing/readme.md](JSONParsing/readme.md)

---

## 6. Core Concepts Used Everywhere

### The `Document` object

```python
from langchain_core.documents import Document

doc = Document(
    page_content="Transformers rely entirely on attention mechanisms...",
    metadata={"source": "pdf/attention.pdf", "page": 1, "total_pages": 15},
)
```

- **`page_content`** is what gets embedded and searched, and what the LLM finally reads.
- **`metadata`** is never embedded, but it is stored next to the vector. It is used for **citations** ("page 3 of attention.pdf"), **filtering** (only search one file, one sheet or one category), **context** for the LLM, and **debugging** bad chunks.

### Document Loaders

Every LangChain loader shares one interface:

```python
docs: list[Document] = loader.load()   # or loader.lazy_load() to stream large inputs
```

Because the interface is the same, you can swap one loader for another without touching the rest of the pipeline.

### Text Splitters, Chunk Size and Overlap

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # max characters per chunk
    chunk_overlap=200,     # characters repeated between neighbouring chunks
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(docs)   # metadata is copied to every chunk
```

| Setting | Too small | Too large | Typical starting point |
|---|---|---|---|
| `chunk_size` | Chunks lose context, so the answer is spread across many of them | Chunks mix several topics, which blurs the embedding and wastes prompt space | 500–1000 characters (≈ 100–250 tokens) |
| `chunk_overlap` | Sentences cut at a boundary lose meaning | Duplicate content, bigger index | 10–20% of `chunk_size` |

**Rules of thumb:**
- Prose → `RecursiveCharacterTextSplitter`.
- Strict model token limits → `TokenTextSplitter` (or `from_tiktoken_encoder`).
- Structured sources (tables, JSON records, Word elements) → often **no splitting at all**: one record = one Document, and only split records that are longer than your chunk size.
- Always check a few chunks by eye before you embed anything.

### Granularity: what is "one Document"?

| Source | Natural unit used in this repo |
|---|---|
| Text file | Whole file, then split into chunks |
| PDF | One page, then split into chunks |
| Word | Whole file (`Docx2txtLoader`) or one element (`Unstructured`) |
| CSV | One row |
| Excel | One sheet (or one row for large sheets) |
| JSON | One record (e.g. one employee, one event) |

---

## 7. Choosing a Loader: Cross-Format Comparison

| Format | Quick start | When you need more control / quality | Install |
|---|---|---|---|
| `.txt` | `TextLoader` | `DirectoryLoader` for folders | built in |
| `.pdf` | `PyPDFLoader` | `PyMuPDFLoader` (speed, metadata), `UnstructuredPDFLoader` (tables, scans, OCR) | `pypdf`, `pymupdf`, `unstructured[pdf]` |
| `.docx` | `Docx2txtLoader` | `UnstructuredWordDocumentLoader(mode="elements")` | `docx2txt`, `unstructured`, `python-docx` |
| `.csv` | `CSVLoader` | Custom pandas function | `pandas` |
| `.xlsx` | Custom pandas function (per sheet) | `UnstructuredExcelLoader` | `openpyxl`, `unstructured[xlsx]` |
| `.json` / `.jsonl` | `JSONLoader` + `jq_schema` | Custom Python flattening | `jq` |

**General rule:** start with the simplest LangChain loader to get a baseline. Move to a structure-aware loader or your own parsing code once you know which questions users ask and where retrieval goes wrong.

---

## 8. Setup

### Prerequisites

- **Python 3.13** (the notebooks were run on CPython 3.13.14; 3.10+ should work for most of the code)
- **Git**
- *Optional:* [uv](https://docs.astral.sh/uv/) for fast environment management (the existing `.venv` was created with uv)
- *Optional:* **VS Code** with the *Python* and *Jupyter* extensions
- *Optional:* **LibreOffice** (only needed to load legacy `.doc` files with Unstructured)
- *Optional:* **Tesseract / Poppler** (only needed for OCR on scanned PDFs with `unstructured[pdf]`)

### 1. Clone the repository

```powershell
git clone <repository-url> RAG
cd RAG
```

### 2. Create a virtual environment (skip if `.venv` already exists)

```powershell
# with the standard library
python -m venv .venv

# or with uv
uv venv .venv --python 3.13
```

### 3. Activate it

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Git Bash
source .venv/Scripts/activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install the dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# or, with uv
uv pip install -r requirements.txt
```

Optional extras for the Unstructured loaders:

```powershell
python -m pip install "unstructured[pdf]"    # UnstructuredPDFLoader (layout, OCR)
python -m pip install "unstructured[xlsx]"   # UnstructuredExcelLoader
python -m pip install "unstructured[docx]"   # UnstructuredWordDocumentLoader
```

### 5. Add your API keys (needed for the later stages)

The current lessons run **fully offline and need no API keys**. The planned embedding and generation stages use hosted models, so create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your-groq-key
OPENAI_API_KEY=your-openai-key
```

and load it in a notebook with:

```python
from dotenv import load_dotenv
load_dotenv()
```

`.env` is listed in [.gitignore](.gitignore), so your keys are never committed. **Never paste keys directly into a notebook**, because notebook outputs and source are saved in the `.ipynb` file.

### 6. Start Jupyter

**Option A: browser**

```powershell
python -m notebook
```

Then open any `*/code.ipynb`.

**Option B: VS Code**

Open a `code.ipynb`, click **Select Kernel** (top right) → **Python Environments** → `.venv`.

### 7. Run the notebooks

- Run the cells **from top to bottom**. Several notebooks *create* their own sample files in early cells (e.g. `products.csv`, `inventory.xlsx`, `company_data.json`, `events.jsonl`, the `.txt` files), and later cells read them.
- All paths are **relative to the notebook's folder** (`pdf/attention.pdf`, `docx/proposal.docx`, …). Jupyter and VS Code start the kernel in the notebook's folder by default, so this works as long as you don't change the working directory.

---

## 9. Dependencies Explained

All packages are listed in [requirements.txt](requirements.txt).

### Used by the current lessons

| Package | Purpose | Used in |
|---|---|---|
| `langchain` | Core framework (pulls in `langchain-core` and `langchain-text-splitters`) | All |
| `langchain-community` | Community document loaders (`TextLoader`, `PyPDFLoader`, `CSVLoader`, `JSONLoader`, …) | All |
| `tiktoken` | OpenAI's tokenizer, used by `TokenTextSplitter` | DataIngestion |
| `pypdf` | Pure-Python PDF reader behind `PyPDFLoader` | DataParsing |
| `pymupdf` | Fast C-based PDF engine behind `PyMuPDFLoader` | DataParsing |
| `pdfminer` | PDF text extraction backend used by Unstructured | DataParsing |
| `unstructured` | Element-based parsing for PDF, Word, Excel, CSV | DataParsing, ParsingWordDoc, ParsingCSV |
| `python-docx` | Reads `.docx` structure (used by Unstructured) | ParsingWordDoc |
| `docx2txt` | Plain-text extraction behind `Docx2txtLoader` | ParsingWordDoc |
| `pandas` | DataFrames for CSV/Excel processing | DataIngestion, ParsingCSV |
| `openpyxl` | Excel `.xlsx` engine for pandas | ParsingCSV |
| `jq` | Python bindings for jq, required by `JSONLoader` | JSONParsing |
| `jupyter` | Notebook server | All |
| `matplotlib` | Plotting (e.g. chunk-length distributions) | Optional |

### Installed for the upcoming stages

| Package | Purpose | Stage |
|---|---|---|
| `sentence-transformers`, `langchain-huggingface` | Free, local embedding models (e.g. `all-MiniLM-L6-v2`) | Embeddings |
| `langchain-openai` | OpenAI embeddings and chat models | Embeddings / Generation |
| `faiss-cpu` | Facebook AI Similarity Search: fast in-memory vector index | Vector store |
| `chromadb` | Persistent vector database with metadata filtering | Vector store |
| `rank_bm25` | Keyword (BM25) ranking for hybrid search | Retrieval |
| `langchain-experimental` | Experimental utilities, e.g. `SemanticChunker` | Chunking / Retrieval |
| `langchain-groq` | Very fast hosted open-weight LLMs through the Groq API | Generation |
| `wikipedia` | Wikipedia loader / retriever as an external knowledge source | Retrieval |
| `langgraph` | Graph-based agent workflows for agentic / multi-step RAG | Agentic RAG |
| `python-dotenv` | Loads API keys from `.env` | All API-based stages |

---

## 10. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `ModuleNotFoundError` inside Jupyter even though the package is installed | The notebook uses a different kernel/Python | Select the `.venv` kernel, or run `python -m pip install ...` from inside the activated venv |
| `FileNotFoundError` for `pdf/attention.pdf`, `files/...` | The working directory is not the notebook's folder, or the sample files have not been generated yet | Open the notebook from its own folder and run the cells in order from the top |
| `UnicodeDecodeError` when loading text on Windows | Windows' default encoding is not UTF-8 | `TextLoader(path, encoding="utf-8")` or `loader_kwargs={"encoding": "utf-8"}` |
| `pip install jq` tries to compile and fails | pip is too old to find the prebuilt wheel | `python -m pip install --upgrade pip`, then retry |
| `UnstructuredExcelLoader` / `UnstructuredPDFLoader` raise import errors | The base `unstructured` package lacks format extras | `pip install "unstructured[xlsx]"` / `"unstructured[pdf]"` |
| Unstructured prints a promotional notice | Default behaviour | `$env:UNSTRUCTURED_DISABLE_NOTICE = "1"` |
| `LangChainDeprecationWarning` on import | Some loaders are moving to standalone packages | Safe to ignore. The loaders still work. |
| Legacy `.doc` files fail to load | Unstructured converts `.doc` via LibreOffice | Install LibreOffice, or save the file as `.docx` |
| PDF text has words glued together, `ﬁ`/`ﬂ` characters, or one word per line | Normal PDF extraction artefacts | Use the cleaning step from [DataParsing](DataParsing/readme.md#3-handling-pdf-challenges-text-cleaning) or try `PyMuPDFLoader` |
| Paths work on Windows but not on Linux/macOS | Linux file systems are case-sensitive (`ParsingCSV/Files/` vs `files/`) | Make the path in the code match the folder name's exact case |

---

## 11. Roadmap

- [x] Documents, text loaders and splitters
- [x] PDF parsing, cleaning and a reusable processing class
- [x] Word, CSV/Excel and JSON/JSONL parsing
- [ ] **Embeddings:** HuggingFace `sentence-transformers` vs OpenAI, cosine similarity, semantic chunking
- [ ] **Vector stores:** FAISS and Chroma, persistence, metadata filtering
- [ ] **Retrieval:** similarity search, MMR, hybrid search (BM25 + vectors), re-ranking
- [ ] **Generation:** prompt templates, LCEL chains, citations, Groq and OpenAI chat models
- [ ] **Evaluation:** measuring retrieval and answer quality
- [ ] **Agentic RAG:** multi-step retrieval and tool use with LangGraph

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation: answering with an LLM using text retrieved from your own data |
| **LLM** | Large Language Model, e.g. GPT, Llama, Claude |
| **Document** | LangChain's unit of data: `page_content` + `metadata` |
| **Metadata** | Data about the data: source, page, row, sheet, category, … |
| **Loader** | A class that reads a source and returns `List[Document]` |
| **Parsing** | Converting a raw file format into plain text + structure |
| **Chunk** | A small piece of a document, sized for embedding and for the prompt |
| **Chunk size / overlap** | Maximum chunk length / amount of text shared by neighbouring chunks |
| **Separator** | The string a splitter prefers to cut on (`"\n\n"`, `"\n"`, `" "`) |
| **Token** | The unit a model reads, about ¾ of an English word on average |
| **Embedding** | A vector of numbers that represents the meaning of a text |
| **Vector store / vector DB** | A database that stores embeddings and finds the nearest ones to a query |
| **Similarity search** | Finding the chunks whose embeddings are closest to the question's (e.g. by cosine similarity) |
| **Top-k** | The number of chunks retrieved per question |
| **BM25** | A classic keyword-based ranking algorithm, often combined with vectors in *hybrid search* |
| **Hallucination** | A fluent but made-up answer from an LLM |
| **Grounding** | Making an answer depend on, and cite, retrieved source text |
| **Ligature** | A single glyph for several letters (`ﬁ`, `ﬂ`) that PDFs often produce |
| **OCR** | Optical Character Recognition: reading text from images or scanned pages |
| **jq** | A small query language for selecting parts of JSON |
| **JSONL** | JSON Lines: one JSON object per line |
| **Element** | A typed piece of a document in Unstructured (`Title`, `NarrativeText`, `Table`, …) |

Each module's README has its own, more detailed glossary.

---

## 13. References

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020): <https://arxiv.org/abs/2005.11401>
- Vaswani et al., *Attention Is All You Need* (2017), the sample PDF: <https://arxiv.org/abs/1706.03762>
- LangChain documentation: <https://python.langchain.com/>
- LangChain document loaders: <https://python.langchain.com/docs/integrations/document_loaders/>
- LangChain text splitters: <https://python.langchain.com/docs/concepts/text_splitters/>
- Unstructured documentation: <https://docs.unstructured.io/>
- jq manual: <https://jqlang.github.io/jq/manual/>
