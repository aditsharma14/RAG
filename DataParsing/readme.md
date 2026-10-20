# Data Parsing for RAG — PDF Loading, Cleaning & Chunking

This folder covers the **first stage of a RAG pipeline**: getting text *out* of PDF files, cleaning it, and splitting it into chunks that can later be embedded and stored in a vector database.

```
DataParsing/
├── code.ipynb        # Notebook with all the experiments
├── pdf/
│   └── attention.pdf # Sample input: "Attention Is All You Need" (Vaswani et al.), 15 pages
└── readme.md         # This file
```

---

## 1. Core Concepts

### RAG (Retrieval-Augmented Generation)
**Definition:** A technique where a Large Language Model (LLM) answers questions using information *retrieved* from your own documents, rather than relying only on what it learned during training.

**Explanation:** A RAG pipeline usually has these stages:

```
PDF / files ──► Parse ──► Clean ──► Chunk ──► Embed ──► Vector DB ──► Retrieve ──► LLM answer
               └──────── this folder ────────┘
```

If parsing is poor (broken words, missing text, junk headers), every later stage gets worse — the retriever can't find what isn't extracted correctly. That's why data parsing matters so much.

### Data Parsing
**Definition:** Converting raw files (PDF, Word, HTML, etc.) into plain, structured text plus metadata that a program can work with.

**Explanation:** PDFs are made for *display*, not for *reading by machines*. Internally a PDF stores positioned characters and glyphs, not paragraphs. A parser has to rebuild the reading order, which often leads to odd line breaks, extra spaces and merged characters.

### Document (LangChain)
**Definition:** `langchain_core.documents.Document` — LangChain's basic unit of data. It has two fields:
- `page_content` — the text (a `str`)
- `metadata` — a `dict` of extra information (source file, page number, etc.)

**Explanation:** Every loader returns a `List[Document]`. For PDFs, loaders usually create **one Document per page**. Text splitters then turn those into smaller Documents (chunks), copying the metadata to each chunk so you always know where a chunk came from.

### Metadata
**Definition:** Data about the data — information describing a document rather than its content.

**Explanation:** For PDFs this includes the file path (`source`), page number (`page`), total pages, creation date, producer software, and more. It lets you:
- cite sources in answers ("found on page 3 of attention.pdf"),
- filter retrieval (e.g. only search one file or a date range),
- debug where bad chunks came from.

### Document Loader
**Definition:** A LangChain class that reads a source (file, URL, database) and returns `Document` objects. Every loader shares the same interface: `loader.load() -> List[Document]`.

---

## 2. PDF Loaders Used in the Notebook

All three are imported from `langchain_community.document_loaders`:

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader
)
```

### 2.1 PyPDFLoader
**Definition:** A loader built on the pure-Python **`pypdf`** library.

**Usage:**
```python
pypdf_loader = PyPDFLoader("pdf/attention.pdf")
pypdf_docs = pypdf_loader.load()
```

**Result on `attention.pdf`:**
- Loaded **15 pages** → 15 Documents
- Page 1 begins: `"Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and..."`
- Metadata example:
  ```python
  {'producer': 'pdfTeX-1.40.25', 'creator': 'LaTeX with hyperref',
   'creationdate': '2024-04-10T21:11:43+00:00', 'source': 'pdf/attention.pdf',
   'total_pages': 15, 'page': 0, 'page_label': '1', ...}
  ```

**Explanation:**
- `page` is **0-indexed** (first page = 0); `page_label` is the label printed on the page ("1").
- **Pros:** no system dependencies, easy to install, reliable for simple text PDFs.
- **Cons:** slower, and less accurate with complex layouts (multi-column pages, tables).

### 2.2 PyMuPDFLoader
**Definition:** A loader built on **PyMuPDF** (`fitz`), a Python binding for the MuPDF C library.

**Usage:**
```python
pymupdf_loader = PyMuPDFLoader("pdf/attention.pdf")
pymupdf_docs = pymupdf_loader.load()
```

**Result on `attention.pdf`:** also **15 pages**, same first-page text, but **richer metadata**:
```python
{'producer': 'pdfTeX-1.40.25', 'creator': 'LaTeX with hyperref',
 'source': 'pdf/attention.pdf', 'file_path': 'pdf/attention.pdf',
 'total_pages': 15, 'format': 'PDF 1.5',
 'modDate': 'D:20240410211143Z', 'creationDate': 'D:20240410211143Z', 'page': 0, ...}
```

**Explanation:**
- **Pros:** very fast (C backend), good text quality, extra fields such as `format` and `file_path`.
- **Cons:** larger dependency; PyMuPDF uses the AGPL license, which matters for commercial use.

### 2.3 UnstructuredPDFLoader
**Definition:** A loader built on the **`unstructured`** library. It detects document *elements* such as titles, narrative text, list items and tables.

**Explanation:** It's imported in the notebook but not used yet. It's the best option when you need layout awareness (e.g. keeping titles apart from body text, or extracting tables). It is heavier: it may need system tools like `poppler` and `tesseract` for OCR on scanned PDFs.

### Loader Comparison

| Loader | Backend | Speed | Metadata | Best for |
|---|---|---|---|---|
| `PyPDFLoader` | pypdf (pure Python) | Medium | Standard | Simple text PDFs, easy setup |
| `PyMuPDFLoader` | MuPDF (C) | Fast | Rich | Large batches, general use |
| `UnstructuredPDFLoader` | unstructured | Slow | Element-level | Complex layouts, tables, scanned docs (OCR) |

> **Observation from the output:** both loaders turn **Figure 5** of the paper (the attention visualisation) into one word per line (`The\nLaw\nwill\nnever\nbe\nperfect...`). Text that sits inside figures is extracted as noise. This is a common PDF problem, and it's one reason a cleaning step is needed.

---

## 3. Handling PDF Challenges (Text Cleaning)

### Common problems in extracted PDF text

| Problem | Example | Cause |
|---|---|---|
| **Extra whitespace / newlines** | `"\n\n\n    The financial..."` | PDF line positions converted into `\n` and spaces |
| **Ligatures** | `ﬁnancial`, `workﬂow` | Fonts combine letter pairs into a single glyph |
| **Headers / footers** | `"Page 1 of 10"` | Repeated on every page, adds noise |
| **Broken words** | `optimi-\nzation` | Hyphenation at line ends |
| **Figure / table text** | one word per line | Text placed inside images or diagrams |

### Ligature
**Definition:** A single typographic character made from two or more letters joined together, e.g. `ﬁ` (U+FB01) for "fi" and `ﬂ` (U+FB02) for "fl".

**Explanation:** It looks right to a human, but to a computer `"ﬁnancial"` ≠ `"financial"`. Keyword search, tokenizers and embeddings may all treat it as a different word, which hurts retrieval. Replacing ligatures with normal letters fixes this.

### The `clean_text` function

```python
def clean_text(text):
    text = " ".join(text.split())   # collapse all whitespace
    text = text.replace("ﬁ", "fi")  # fix fi ligature
    text = text.replace("ﬂ", "fl")  # fix fl ligature
    return text
```

**How it works:**
1. `text.split()` with no argument splits on **any** whitespace (spaces, tabs, `\n`) and drops empty pieces.
2. `" ".join(...)` puts the words back together with a single space, so all extra newlines and indentation disappear.
3. `.replace()` swaps each ligature for its normal letters.

**Before:**
```
"Company Financial Report\n\n\n    The ﬁnancial performance for ﬁscal year 2024\n    shows signiﬁcant growth in proﬁtability.\n\n\n\n    Revenue increased by 25%.\n\nThe company's efﬁciency improved due to workﬂow\noptimization.\n\n\nPage 1 of 10\n"
```
**After:**
```
"Company Financial Report The financial performance for fiscal year 2024 shows significant growth in profitability. Revenue increased by 25%. The company's efficiency improved due to workflow optimization. Page 1 of 10"
```

> **Trade-off:** collapsing all whitespace also removes paragraph breaks, which some splitters use to find natural boundaries. `"Page 1 of 10"` is also still there. Removing it would need an extra regex, e.g. `re.sub(r"Page \d+ of \d+", "", text)`.

---

## 4. Chunking

### Chunking (Text Splitting)
**Definition:** Breaking long text into smaller, overlapping pieces called **chunks**.

**Why it's needed:**
- **Embedding models have input limits**: they can only embed a certain number of tokens at a time.
- **Precision**: a small, focused chunk matches a question better than an entire page.
- **LLM context window**: only the top-k relevant chunks are sent to the LLM, not whole documents.

### Chunk Size
**Definition:** The maximum length of each chunk, measured in **characters** by default in LangChain.

**Explanation:** If it's too small, chunks lose context ("it increased by 25%" — what did?). If it's too large, chunks mix several topics and retrieval becomes fuzzy. 500–1500 characters is a common starting range.

### Chunk Overlap
**Definition:** The number of characters shared between the end of one chunk and the start of the next.

**Explanation:** Overlap stops a sentence or idea from being cut in half at a chunk boundary and lost. Example with `chunk_size=1000, chunk_overlap=100`: chunk 1 = chars 0–1000, chunk 2 ≈ chars 900–1900, and so on.

### RecursiveCharacterTextSplitter
**Definition:** LangChain's recommended general-purpose splitter (`from langchain_text_splitters import RecursiveCharacterTextSplitter`).

**How it works:** It tries a list of **separators** in order, by default `["\n\n", "\n", " ", ""]`. It splits on the first separator. If a piece is still larger than `chunk_size`, it splits that piece again using the next separator, and so on. This keeps paragraphs whole when possible, then sentences or lines, then words.

**Key methods:**
- `split_documents(docs)`: splits existing `Document`s.
- `create_documents(texts=[...], metadatas=[...])`: builds `Document`s from raw strings and attaches the given metadata to every chunk. The notebook uses this one.

---

## 5. `SmartPDFProcessor` — the full pipeline class

**Definition:** A custom class in the notebook that combines **load → clean → filter → chunk → enrich metadata** in one call.

```python
processor = SmartPDFProcessor()                     # chunk_size=1000, chunk_overlap=100
smart_chunks = processor.process_pdf("pdf/attention.pdf")
```

### Components

| Part | What it does |
|---|---|
| `__init__(chunk_size=1000, chunk_overlap=100)` | Stores the settings and creates a `RecursiveCharacterTextSplitter` with `separators=[" "]` (splits on spaces only) |
| `process_pdf(pdf_path) -> List[Document]` | The main pipeline, described below |
| `_clean_text(text)` | Private helper (the leading `_` is a naming convention). Same logic as `clean_text`: collapses whitespace and fixes ligatures |

### Step by step: `process_pdf`
1. **Load:** `PyPDFLoader(pdf_path).load()` returns one Document per page.
2. **Loop over pages** with `enumerate(pages)` to get `page_num`.
3. **Clean** each page using `_clean_text`.
4. **Skip nearly empty pages:** pages with fewer than 50 characters (blank pages, pages with only an image) are dropped so they don't create useless chunks.
5. **Chunk + enrich metadata** with `create_documents`. Each chunk keeps the original PDF metadata (`**page.metadata`) and gets these extra fields:
   - `page`: `page_num + 1`, overriding the loader's 0-indexed value with a **1-indexed** page number
   - `total_pages`: number of pages in the PDF
   - `chunk_method`: `"smart_pdf_processor"`, which records how the chunk was made
   - `char_count`: length of the cleaned page text
6. **Collect** all chunks with `processed_chunks.extend(chunks)` and return them.

### Result on `attention.pdf`
```
Processed into 49 smart chunks

Sample chunk metadata:
  producer: pdfTeX-1.40.25
  creator: LaTeX with hyperref
  source: pdf/attention.pdf
  total_pages: 15
  page: 1
  page_label: 1
  chunk_method: smart_pdf_processor
  char_count: 2857
  ...
```
15 pages → **49 chunks** (about 3 chunks per page at 1000 characters each).

### Things to know / possible improvements
- **Trailing commas bug:** `self.chunk_size=chunk_size,` and `self.chunk_overlap=chunk_overlap,` end with commas, so they store **tuples** (`(1000,)`), not integers. The splitter isn't affected because it gets the raw arguments, but any later code that reads `self.chunk_size` will get a tuple. Remove the commas to fix this.
- **`separators=[" "]`:** splits on spaces only. That's fine here, since `_clean_text` already removed all newlines, but it means chunks can end mid-sentence. Splitting on sentence boundaries (`[". ", " ", ""]`) usually gives cleaner chunks.
- **Chunks never cross pages:** each page is chunked separately, so a paragraph that continues onto the next page is split at the page break.
- **`char_count`** is the length of the whole *page*, not of each chunk.
- `UnstructuredPDFLoader` is imported but not used yet. It's worth trying for table-heavy PDFs.

---

## 6. Setup

```bash
pip install langchain-community langchain-text-splitters langchain-core pypdf pymupdf
# optional, for UnstructuredPDFLoader:
pip install "unstructured[pdf]"
```

Then open `code.ipynb` and run the cells from top to bottom. The notebook expects the PDF at `pdf/attention.pdf`, relative to this folder.

---

## 7. Quick Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation: an LLM answers using documents retrieved from your own data |
| **Parsing** | Pulling usable text and metadata out of raw files |
| **Document** | LangChain object with `page_content` (text) and `metadata` (dict) |
| **Loader** | Class that reads a source and returns `List[Document]` via `.load()` |
| **Metadata** | Information about a document: source, page, dates, etc. |
| **Ligature** | Combined glyph such as `ﬁ` / `ﬂ` that must be normalised to `fi` / `fl` |
| **Cleaning** | Removing whitespace noise and fixing character problems in extracted text |
| **Chunk** | A small piece of a document used for embedding and retrieval |
| **Chunk size** | Maximum characters per chunk |
| **Chunk overlap** | Characters repeated between neighbouring chunks to keep context |
| **Separator** | String a splitter uses as a split point (`"\n\n"`, `"\n"`, `" "`, ...) |
| **RecursiveCharacterTextSplitter** | Splitter that tries separators in order, from coarse to fine |
| **Embedding** | *(next stage)* A numeric vector representing the meaning of a chunk |
| **Vector DB** | *(next stage)* A database that stores embeddings for similarity search |
