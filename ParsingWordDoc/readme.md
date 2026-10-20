# Parsing Word Documents for RAG — Docx2txtLoader vs UnstructuredWordDocumentLoader

This folder covers loading **Microsoft Word (`.docx`) files** into LangChain `Document` objects, the first stage of a RAG pipeline. The notebook compares two LangChain loaders on the same sample file:

1. **`Docx2txtLoader`**: pulls out all the text as one plain string.
2. **`UnstructuredWordDocumentLoader`**: splits the file into typed **elements** (Title, NarrativeText, ListItem, Table, …).

```
ParsingWordDoc/
├── code.ipynb            # Notebook with both loading methods
├── docx/
│   ├── proposal.docx     # Sample input: a short "RAG Implementation" project proposal
│   └── proposal.txt      # Plain-text version of the same proposal (for reference)
└── readme.md             # This file
```

---

## 1. Core Concepts

### RAG (Retrieval-Augmented Generation)
**Definition:** A technique where a Large Language Model (LLM) answers questions using information *retrieved* from your own documents, rather than relying only on what it learned during training.

**Explanation:** A RAG pipeline usually has these stages:

```
.docx / files ──► Parse ──► Chunk ──► Embed ──► Vector DB ──► Retrieve ──► LLM answer
                 └ this folder ┘
```

Everything downstream depends on parsing. If headings get lost or table cells get jumbled, the retriever has less to work with.

### What is a `.docx` file?
**Definition:** The Office Open XML format used by Microsoft Word since 2007. A `.docx` is a **ZIP archive** that holds XML files. The main content lives in `word/document.xml`.

**Explanation:** A PDF only stores characters at positions on a page. A `.docx` keeps the logical structure: paragraphs (`<w:p>`), text runs (`<w:r>`), named styles such as `Heading 1` and `List Bullet`, tables (`<w:tbl>`), and so on. Because that structure is in the file, Word documents are usually **easier and more accurate to parse than PDFs**: there's no reading order to rebuild and no broken ligatures to fix. A good parser can also use the styles to tell a heading from body text.

You can check this yourself: rename `proposal.docx` to `proposal.zip`, open it, and look inside `word/document.xml`.

### Document (LangChain)
**Definition:** `langchain_core.documents.Document` is LangChain's basic unit of data. It has two fields:
- `page_content`: the text (a `str`)
- `metadata`: a `dict` of extra information (source file, category, etc.)

**Explanation:** Every loader returns a `List[Document]`. Word files have no fixed pages, so the number of Documents you get depends on the loader. `Docx2txtLoader` returns **1** Document for the whole file. `UnstructuredWordDocumentLoader` in `"elements"` mode returns **one Document per element**, which is 20 for `proposal.docx`.

### Document Loader
**Definition:** A LangChain class that reads a source (file, URL, database) and returns `Document` objects. Every loader has the same interface: `loader.load() -> List[Document]`. That means you can swap one loader for another without changing the rest of the pipeline.

### Metadata
**Definition:** Data about the data, meaning information that describes a document rather than its content.

**Explanation:** Metadata lets you cite sources, filter during retrieval (for example, "only search Title elements" or "only this file"), and track down where a bad chunk came from. The two loaders in this notebook produce very different amounts of metadata (see the comparison in section 4).

---

## 2. Libraries Used

| Library | Role in this folder |
|---|---|
| **`langchain-community`** | Provides both loader classes in `langchain_community.document_loaders`. |
| **`langchain-core`** | Defines the `Document` class the loaders return (installed with `langchain`). |
| **`docx2txt`** | A small pure-Python library that unzips the `.docx`, reads its XML and returns the plain text. `Docx2txtLoader` uses it under the hood. |
| **`unstructured`** | A document-partitioning library. Its `partition_docx` function breaks a Word file into typed elements. `UnstructuredWordDocumentLoader` uses it under the hood. |
| **`python-docx`** | The library `unstructured` uses to read `.docx` paragraphs, styles and tables. |

All of these are listed in the project's root [requirements.txt](../requirements.txt).

> **Note:** When the notebook imports from `langchain_community`, it prints a `DeprecationWarning` saying the package is being sunset in favour of standalone integration packages. The loaders still work. The warning just means future code should move to the dedicated packages (e.g. `langchain-unstructured`) once they're available.

---

## 3. The Code, Cell by Cell

### Cell 1: Imports

```python
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
```

This imports both loader classes. Nothing is read from disk yet. A loader only does work when you call `.load()`.

### Cell 2: Method 1, `Docx2txtLoader`

```python
print("1️⃣ Using Docx2txtLoader")
try:
    docx_loader = Docx2txtLoader("docx/proposal.docx")
    docs = docx_loader.load()
    print(f"✅ Loaded {len(docs)} document(s)")
    print(f"Content preview: {docs[0].page_content[:200]}...")
    print(f"Metadata: {docs[0].metadata}")
except Exception as e:
    print(f"Error: {e}")
```

**What it does:**
1. `Docx2txtLoader("docx/proposal.docx")` creates the loader. The path is relative to the notebook's folder.
2. `.load()` calls `docx2txt.process(path)`, which unzips the file, walks `document.xml`, and joins all text with newlines.
3. The entire file becomes **one** `Document`.
4. `page_content[:200]` prints the first 200 characters as a preview.
5. The `try/except` keeps the notebook running if the file is missing or corrupt, and prints the error instead.

**Output:**
```
✅ Loaded 1 document(s)
Content preview: Project Proposal: RAG Implementation

Executive Summary

This proposal outlines the implementation of a Retrieval-Augmented Generation system for our organization.

Objectives

Key objectives include:...
Metadata: {'source': 'docx/proposal.docx'}
```

**Takeaways:**
- Fast, with very few dependencies.
- The output is **flat text**. Headings, list items and body text all look the same, and table cells come out as text separated by newlines.
- The only metadata is `source`.
- Before embedding, you'd normally pass this one big Document to a text splitter (e.g. `RecursiveCharacterTextSplitter`).

### Cell 3: Method 2, `UnstructuredWordDocumentLoader`

```python
print("\n2️⃣ Using UnstructuredWordDocumentLoader")
try:
    unstructured_loader = UnstructuredWordDocumentLoader("docx/proposal.docx", mode="elements")
    unstructured_docs = unstructured_loader.load()

    print(f"✅ Loaded {len(unstructured_docs)} elements")
    for i, doc in enumerate(unstructured_docs[:3]):
        print(f"\nElement {i+1}:")
        print(f"Type: {doc.metadata.get('category', 'unknown')}")
        print(f"Content: {doc.page_content[:100]}...")
except Exception as e:
    print(e)
```

**What it does:**
1. `mode="elements"` tells the loader to return **one Document per element** instead of merging everything.
2. `.load()` calls `unstructured.partition.docx.partition_docx`. It reads each paragraph and table, uses the Word style and the text itself to classify it, and attaches rich metadata.
3. The loop prints the first 3 elements with their `category`. `.get('category', 'unknown')` avoids a `KeyError` if the key is missing.

**Output:**
```
✅ Loaded 20 elements

Element 1:
Type: Title
Content: Project Proposal: RAG Implementation...

Element 2:
Type: Title
Content: Executive Summary...

Element 3:
Type: NarrativeText
Content: This proposal outlines the implementation of a Retrieval-Augmented Generation system for our organiz...
```

(It also prints a promotional notice from Unstructured. You can hide it by setting the environment variable `UNSTRUCTURED_DISABLE_NOTICE=1`.)

#### The `mode` parameter

| `mode` | Result |
|---|---|
| `"single"` (default) | All elements joined into **one** Document, which is similar to `Docx2txtLoader`. |
| `"elements"` | **One Document per element**, each with its own `category` and metadata. *(Used here.)* |
| `"paged"` | One Document per page. Word files rarely have reliable page info, so this is not very useful for `.docx`. |

### Cell 4: Inspecting all elements

```python
unstructured_docs
```

This displays the full list of 20 Documents. The `proposal.docx` file breaks down like this:

| # | Category | Content |
|---|---|---|
| 1 | Title | Project Proposal: RAG Implementation |
| 2 | Title | Executive Summary |
| 3 | NarrativeText | This proposal outlines the implementation of … |
| 4 | Title | Objectives |
| 5 | NarrativeText | Key objectives include: |
| 6–8 | ListItem | Improve information retrieval accuracy / Reduce response time … / Integrate with existing knowledge base |
| 9 | Title | Budget and Timeline |
| 10–12 | UncategorizedText | Budget: $50,000 / Timeline: 3 months / Team: 4 developers, 1 project manager |
| 13 | Title | Technical Requirements |
| 14 | UncategorizedText | Required technologies: |
| 15–18 | ListItem | Python 3.8+ / OpenAI API access / Vector database (ChromaDB) / 16GB RAM minimum |
| 19 | Title | Project Phases |
| 20 | Table | Phase · Duration · Deliverables · Research · 2 weeks · Technology evaluation … |

The `.docx` includes a **"Project Phases" table** that is not in `proposal.txt`. Unstructured keeps it as a single `Table` element. In `elements` mode, its metadata usually also includes `text_as_html`, which preserves the row and column structure.

#### Example element metadata

```python
{
  'source': 'docx/proposal.docx',
  'filename': 'proposal.docx',
  'file_directory': 'docx',
  'filetype': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'last_modified': '2026-10-20T11:24:48',
  'languages': ['eng'],
  'category': 'NarrativeText',
  'category_depth': 0,
  'element_id': 'bbc04fc71e33a92df30d7fe7c33b6375',
  'parent_id': 'c0f844859abf08d9506856b3aed4a719'
}
```

| Key | Meaning |
|---|---|
| `source` / `filename` / `file_directory` | Where the element came from. |
| `filetype` | The MIME type of a `.docx` file. |
| `last_modified` | The file's modification timestamp. |
| `languages` | The detected language (`eng` = English). |
| `category` | The element type (see below). |
| `category_depth` | The nesting level (e.g. heading level or list indent). `0` = top level. |
| `element_id` | A unique hash ID for this element. |
| `parent_id` | The `element_id` of the element's **parent**, usually the heading it sits under. In this example, it points to the "Executive Summary" title. |

#### Element categories

| Category | Meaning |
|---|---|
| **Title** | A heading or title. |
| **NarrativeText** | A normal sentence or paragraph of prose. |
| **ListItem** | A bullet or numbered list item. |
| **Table** | A whole table, with its text flattened in `page_content` and HTML in `text_as_html`. |
| **UncategorizedText** | Short text that Unstructured couldn't classify confidently, like `Budget: $50,000`. |
| Others | `Header`, `Footer`, `PageBreak`, `Image`, `Address`, `EmailAddress`, … |

---

## 4. Loader Comparison

| | `Docx2txtLoader` | `UnstructuredWordDocumentLoader` (`mode="elements"`) |
|---|---|---|
| Backing library | `docx2txt` | `unstructured` (+ `python-docx`) |
| Output for `proposal.docx` | 1 Document | 20 Documents |
| Structure kept? | No, flat text | Yes. Each element gets a category, and children are linked to parents. |
| Tables | Cells flattened into lines | One `Table` element plus an HTML version |
| Metadata | `source` only | Category, depth, IDs, parent, filetype, language, timestamps |
| Speed / install weight | Very fast, light | Slower, heavy dependencies |
| Best for | Quick text dumps, simple documents | Structure-aware chunking, filtering by element type, documents with tables |

**Rule of thumb:** use `Docx2txtLoader` for prototyping or when you only need the text. Use `UnstructuredWordDocumentLoader` with `mode="elements"` when document structure matters for retrieval quality.

---

## 5. Using the Output in a RAG Pipeline

The elements are usually too small to embed one by one (e.g. `"Objectives"` on its own means little). Common next steps:

- **Group by section:** use `parent_id` or the `Title` elements to merge each heading with its body into one chunk. Each chunk then holds one complete idea and carries the heading as context.
- **Filter:** drop noise categories such as `Header`, `Footer` and `PageBreak` before embedding.
- **Tables:** embed the `text_as_html` (or a Markdown conversion) so the LLM can see rows and columns.
- **Flat text from `Docx2txtLoader`:** split it with `RecursiveCharacterTextSplitter` (see [../DataParsing/readme.md](../DataParsing/readme.md) for chunk size and overlap).
- Then embed the chunks (e.g. with `sentence-transformers` / OpenAI embeddings) and store them in a vector DB such as **ChromaDB** or **FAISS**.

---

## 6. Setup

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt   # includes langchain-community, docx2txt, unstructured, python-docx
python -m notebook
```

Then open `ParsingWordDoc/code.ipynb`. Run the notebook **from inside this folder**, because the code uses the relative path `docx/proposal.docx`.

Optional: hide the Unstructured promo message with

```powershell
$env:UNSTRUCTURED_DISABLE_NOTICE = "1"
```

> `UnstructuredWordDocumentLoader` can also read legacy **`.doc`** files, but it needs **LibreOffice** installed to convert them to `.docx` first. `Docx2txtLoader` supports `.docx` only.

---

## 7. Quick Glossary

| Term | Meaning |
|---|---|
| **.docx** | A ZIP of XML files (Office Open XML) that stores a Word document's content and structure. |
| **Document** | LangChain object with `page_content` + `metadata`. |
| **Document Loader** | A class that turns a file or source into `List[Document]` through `.load()`. |
| **Element** | A typed piece of a document (Title, NarrativeText, ListItem, Table, …) produced by Unstructured. |
| **Partitioning** | Unstructured's word for splitting a file into elements. |
| **`mode`** | How the Unstructured loader groups elements: `single`, `elements` or `paged`. |
| **category** | The element type stored in metadata. |
| **parent_id** | Links an element to its parent heading, which allows hierarchy-aware chunking. |
| **text_as_html** | An HTML version of a table, stored in metadata so its structure isn't lost. |
| **Metadata** | Information about the content (source, type, IDs), used for citations and filtering. |
