# Semantic Chunking for RAG: Splitting Text by Meaning, Not by Length

This folder covers **semantic chunking**, a way of splitting documents into chunks based on *what the sentences mean* rather than on how many characters they contain. The previous folders split text with fixed-size splitters such as `RecursiveCharacterTextSplitter`. Those splitters are fast and predictable, but they don't know where one topic ends and the next begins. Semantic chunking uses **embeddings** and **cosine similarity** (see the [Embeddings](../Embeddings/readme.md) folder) to find those topic boundaries.

The notebook builds the idea up in four steps:

1. **Semantic chunking from scratch**: about 20 lines of plain Python that group neighbouring sentences whose embeddings are similar.
2. **A reusable chunker class**: the same logic wrapped in a class with a `split_documents()` method, so it works with LangChain `Document` objects.
3. **A full RAG pipeline on top of the chunks**: FAISS vector store, retriever and a Groq-hosted LLM, first called by hand and then wired together as an LCEL chain.
4. **LangChain's built-in `SemanticChunker`**: the production-ready version from `langchain_experimental`, run on a text file.

```
Semantic-Chunking/
├── code.ipynb            # Notebook with all the experiments
├── langchain_intro.txt   # Small sample text used by the last cells
└── readme.md             # This file
```

---

## 1. Core Concepts

### Chunking
**Definition:** Cutting a long document into smaller pieces (chunks) before embedding and storing them.

**Why it is needed:**
- Embedding models have an input limit (`all-MiniLM-L6-v2` reads at most 256 tokens of input; anything longer is cut off).
- An embedding of a whole 50-page document is a blurry average of every topic in it, so it matches no single question well.
- LLMs have a limited context window, and you pay per token. You want to send the model only the few paragraphs that answer the question.

So a RAG pipeline looks like this:

```
Files ──► Parse ──► Chunk ──► Embed ──► Vector DB ──► Retrieve ──► LLM answer
                    └ this folder ┘
```

### Fixed-size chunking (the usual approach) and its problem
`CharacterTextSplitter` and `RecursiveCharacterTextSplitter` cut text every *N* characters (with some overlap), trying to break at paragraph or sentence boundaries. They ignore meaning, which causes two kinds of bad chunks:

| Problem | Example |
|---|---|
| **One idea split in two** | A chunk ends with *"To reset your password, open Settings and"* and the next chunk starts with *"click Security → Reset."* Neither chunk answers the question on its own. |
| **Unrelated ideas glued together** | One chunk contains the end of a paragraph about LangChain and the start of a paragraph about the Eiffel Tower. Its embedding is a mix of both and matches neither topic well. |

### Semantic chunking
**Definition:** Splitting text at the points where the **meaning changes**, measured by comparing the embeddings of neighbouring sentences.

**The algorithm in plain words:**

1. Split the text into sentences.
2. Embed every sentence (turn it into a vector).
3. Walk through the sentences in order. For each sentence, compute the cosine similarity with the sentence right before it.
4. If the similarity is **high**, the two sentences talk about the same thing, so keep them in the same chunk.
5. If the similarity is **low**, the topic has changed, so close the current chunk and start a new one.

```
Sentence:    S1 ──── S2 ──── S3 ──── S4 ──── S5
Similarity:     0.83    0.46    0.02    0.38
Threshold 0.7:  keep    SPLIT   SPLIT   SPLIT
Result:      [S1 S2] [S3]    [S4]    [S5]
```

(These are the real numbers from the sample text in this notebook; see Cell 1 below.)

### Cosine similarity (quick recap)
A score that says how closely two vectors point in the same direction:

```
cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

- **1.0**: same meaning
- **around 0**: unrelated
- **negative**: opposite direction (rare for sentence embeddings)

The full explanation with worked examples is in the [Embeddings readme](../Embeddings/readme.md).

### Threshold
The cut-off similarity value that decides "same topic" vs "new topic". It is the main tuning knob of the from-scratch chunker:

| Threshold | Effect | Result |
|---|---|---|
| High (e.g. `0.8`) | Only very similar sentences stay together | Many small, tight chunks |
| Medium (e.g. `0.5`–`0.7`) | Balanced | Medium chunks |
| Low (e.g. `0.3`) | Even loosely related sentences stay together | Few large chunks |

There is no universally correct value. It depends on the embedding model (each model has its own "typical" similarity range) and on your data. Try a few values on real documents and look at the chunks.

### Fixed-size vs semantic chunking

| | Fixed-size (`RecursiveCharacterTextSplitter`) | Semantic (`SemanticChunker` / custom) |
|---|---|---|
| Splits on | Character count | Change in meaning |
| Speed | Very fast, no model needed | Slower, every sentence must be embedded |
| Cost | Free | Embedding cost (free if the model runs locally) |
| Chunk size | Predictable | Varies a lot, from one sentence to many |
| Chunk quality | Can cut ideas in half | Each chunk tends to cover one topic |
| Best for | Uniform text, large volumes, quick prototypes | Documents that jump between topics, FAQs, transcripts, notes |

---

## 2. Use Cases

Semantic chunking pays off when a document **changes topic often** and the topic boundaries do not line up with fixed character counts:

- **Meeting transcripts and podcasts**: speakers move between agenda items without headings. Semantic splitting finds where each discussion starts and ends.
- **FAQ pages and help-center articles**: each question-answer pair becomes its own chunk, so a question retrieves exactly one answer.
- **Personal notes, emails and chat logs**: short, mixed-topic text with no formal structure.
- **News digests and newsletters**: several unrelated stories in one document.
- **Research papers and reports**: keeps a whole argument (claim, evidence, conclusion) together instead of cutting it mid-paragraph.
- **Customer support tickets**: separates the problem description from unrelated small talk or signatures.

It is **less useful** (or not worth the extra cost) when:
- Documents are already well structured with headings. A Markdown or HTML header splitter is cheaper and just as good.
- You are indexing millions of documents and embedding every sentence twice (once for chunking, once for storage) is too slow or expensive.
- Text is code, tables or logs, where "sentences" do not exist in the usual sense.

---

## 3. Libraries Used

| Library | Purpose in this notebook |
|---|---|
| `sentence-transformers` | Downloads and runs the `all-MiniLM-L6-v2` embedding model locally |
| `scikit-learn` | `cosine_similarity` to compare two sentence vectors |
| `numpy` | Arrays that hold the embeddings (imported in Cell 0) |
| `langchain-core` | `Document`, `PromptTemplate`, `StrOutputParser`, `RunnableMap` |
| `langchain-community` | `FAISS` vector store wrapper and `TextLoader` |
| `langchain-huggingface` | `HuggingFaceEmbeddings`, the LangChain wrapper around sentence-transformers |
| `langchain-groq` | `ChatGroq`, the chat model client for Groq's hosted LLMs |
| `langchain-experimental` | `SemanticChunker`, LangChain's built-in semantic splitter |
| `faiss-cpu` | The vector index that does the fast similarity search |
| `python-dotenv` | Loads the Groq API key from a `.env` file |
| `langchain-openai` | Imported but not used (the notebook uses free HuggingFace embeddings and Groq instead) |

---

## 4. The Code, Cell by Cell

### Cell 0: Imports for the from-scratch version

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
```

| Line | What it does |
|---|---|
| `from sentence_transformers import SentenceTransformer` | The class that loads a pre-trained sentence embedding model and turns sentences into vectors. |
| `from sklearn.metrics.pairwise import cosine_similarity` | A ready-made, vectorised cosine similarity function. It takes two **2D** arrays (lists of vectors) and returns a matrix of similarities between every pair. |
| `import numpy as np` | NumPy, the array library. `model.encode()` returns NumPy arrays. It is not called directly in this cell's code but is standard to have around. |

---

### Cell 1: Semantic chunking from scratch

This is the heart of the folder. Read it slowly.

#### Step 1: Load the model and prepare sentences

```python
model=SentenceTransformer('all-MiniLM-L6-v2')
```
Downloads (first run only, about 90 MB, cached in `~/.cache/huggingface`) and loads the **all-MiniLM-L6-v2** model. It is small, fast, runs on a CPU, and turns any sentence into a **384-number vector**. The `Loading weights: 0/103` progress bar and the `HF_TOKEN` warning in the output are normal. The warning only means downloads are rate-limited without a free HuggingFace token.

```python
text="""
LangChain is a framework for building applications with LLMs.
Langchain provides modular abstractions to combine LLMs with tools like OpenAI and Pinecone.
You can create chains, agents, memory, and retrievers.
The Eiffel Tower is located in Paris.
France is a popular tourist destination.
"""
```
A deliberately mixed sample: three sentences about **LangChain** and two about **Paris / France**. A good chunker should notice the topic switch between sentence 3 and sentence 4.

```python
sentences=[s.strip() for s in text.split("\n") if s.strip()]
```
A list comprehension that turns the text into a clean list of sentences:
- `text.split("\n")` cuts the string at every newline. Because the text starts and ends with a newline, this produces some empty strings.
- `if s.strip()` keeps only lines that are not empty after removing whitespace.
- `s.strip()` removes leading/trailing spaces from each kept line.

Result: a list of 5 sentences. (This works here because each sentence is on its own line. Real documents need a proper sentence splitter; see "Limitations" below.)

#### Step 2: Embed every sentence

```python
embeddings=model.encode(sentences)
```
Runs all 5 sentences through the model in one batch. Returns a NumPy array of shape **`(5, 384)`**: one row of 384 numbers per sentence. `embeddings[0]` is the vector for sentence 1, and so on.

#### Step 3: Set up the loop

```python
threshold = 0.7  # control chunk tightness
chunks = []
current_chunk=[sentences[0]]
```

| Line | What it does |
|---|---|
| `threshold = 0.7` | Neighbouring sentences with similarity **≥ 0.7** stay together; anything lower starts a new chunk. |
| `chunks = []` | The finished chunks (each one a single string) are collected here. |
| `current_chunk=[sentences[0]]` | The chunk currently being built. It always starts with the first sentence, because the first sentence has no previous sentence to compare against. |

#### Step 4: Walk through the sentences and decide where to split

```python
for i in range(1, len(sentences)):
```
Loops over sentence indexes `1, 2, 3, 4`. It starts at `1` (not `0`) because each sentence is compared with the one **before** it, `i - 1`.

```python
    sim = cosine_similarity(
        [embeddings[i - 1]],
        [embeddings[i]]
    )[0][0]
```
- `[embeddings[i - 1]]` and `[embeddings[i]]`: each vector is wrapped in a list because `cosine_similarity` expects 2D input (a *list* of vectors), not a single 1D vector.
- The function returns a 1×1 matrix like `[[0.826]]`.
- `[0][0]` pulls out the single number.

```python
    if sim>=threshold:
        current_chunk.append(sentences[i])
```
Similar enough to the previous sentence → same topic → add it to the chunk being built.

```python
    else:
        chunks.append(" ".join(current_chunk))
        current_chunk=[sentences[i]]
```
Not similar enough → topic changed:
1. `" ".join(current_chunk)` glues the sentences of the current chunk into one string separated by spaces, and saves it to `chunks`.
2. `current_chunk=[sentences[i]]` starts a fresh chunk containing only the current sentence.

```python
chunks.append(" ".join(current_chunk))
```
After the loop ends, the last chunk is still sitting in `current_chunk` and was never saved (chunks are only saved when a split happens). This line saves it. Forgetting this line is the classic bug in this algorithm; it silently drops the end of every document.

```python
print("\n📌 Semantic Chunks:")
for idx, chunk in enumerate(chunks):
    print(f"\nChunk {idx+1}:\n{chunk}")
```
Prints each chunk with a 1-based number. `enumerate` gives both the index and the item.

#### What happens on the sample text

The actual similarities between neighbouring sentences with `all-MiniLM-L6-v2`:

| Pair | Similarity | ≥ 0.7? | Decision |
|---|---|---|---|
| S1 *"LangChain is a framework…"* ↔ S2 *"Langchain provides modular abstractions…"* | **0.826** | Yes | Same chunk |
| S2 ↔ S3 *"You can create chains, agents…"* | **0.456** | No | Split |
| S3 ↔ S4 *"The Eiffel Tower is located in Paris."* | **0.022** | No | Split (huge topic change) |
| S4 ↔ S5 *"France is a popular tourist destination."* | **0.378** | No | Split |

Output:
```
Chunk 1: LangChain is a framework for building applications with LLMs. Langchain provides modular abstractions to combine LLMs with tools like OpenAI and Pinecone.
Chunk 2: You can create chains, agents, memory, and retrievers.
Chunk 3: The Eiffel Tower is located in Paris.
Chunk 4: France is a popular tourist destination.
```

**What this teaches about the threshold:** the near-zero similarity between S3 and S4 correctly detects the LangChain → Paris switch. But `0.7` is too strict for this model: S3 is clearly still about LangChain (0.456), and S4/S5 are both about France (0.378), yet they got separated. A threshold around **0.3** would give the "ideal" two chunks here (LangChain / Paris). This is why the threshold must be tuned per model and per dataset.

---

### Cell 2: Imports for the LangChain version

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableMap

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
```

| Import | What it is |
|---|---|
| `Document` | LangChain's standard container for text: `page_content` (the text) plus `metadata` (a dict such as `{"source": "file.pdf", "page": 3}`). Every loader, splitter and vector store speaks `Document`. |
| `PromptTemplate` | A reusable prompt with `{placeholders}` that get filled in at run time. |
| `StrOutputParser` | Takes the LLM's message object and returns just its text as a plain `str`. |
| `RunnableLambda` | Wraps any Python function so it can be used as a step in an LCEL chain (imported, not used). |
| `RunnableMap` | Runs several functions on the same input and returns their results as a dict. Used to build `{"context": ..., "question": ...}` in Cell 9. |
| `FAISS` | LangChain wrapper around Facebook AI Similarity Search, an in-memory vector index. |
| `OpenAIEmbeddings`, `ChatOpenAI` | OpenAI clients. Imported but **not used**; the notebook switches to free alternatives (HuggingFace + Groq). |

The `DeprecationWarning` in the output says `langchain-community` is being sunset. The code still works; in new projects, prefer the standalone package for each integration (for example `langchain-text-splitters`, or the dedicated FAISS integration package once you migrate).

---

### Cell 3: API key

```python
import os
os.environ["GROQ_API_KEY"]=os.getenv("groq_api")
```
Reads an environment variable called `groq_api` and copies it into `GROQ_API_KEY`, the name `ChatGroq` looks for.

> **Watch out:** this cell runs *before* `load_dotenv()` (which is only called in Cell 7). If `groq_api` is only defined in your `.env` file and not in your system environment, `os.getenv` returns `None`, and assigning `None` to `os.environ` raises `TypeError: str expected, not NoneType`. Either set the variable in your shell, or run `from dotenv import load_dotenv; load_dotenv()` first.

---

### Cell 4: A reusable semantic chunker class

The logic of Cell 1, packaged as a class so it can be reused and plugged into a LangChain pipeline.

```python
class ThresholdSematicChunker:
    def __init__(self,model_name="all-MiniLM-L6-v2",threshold=0.7):
        self.model=SentenceTransformer(model_name)
        self.threshold=threshold
```
- The constructor takes the model name and threshold as parameters (with sensible defaults), so you can try `ThresholdSematicChunker(threshold=0.4)` or a different model without editing the code.
- The model is loaded **once** when the object is created and stored on `self.model`, so splitting many documents does not reload it each time.
- (The class name has a small typo, "Sematic" instead of "Semantic". It works fine, but keep the spelling consistent if you use it.)

#### `split()`: split one string into chunks

```python
    def split(self, text: str):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
```
**Different from Cell 1:** sentences are found by splitting on **periods** (`.`) instead of newlines. This works on normal paragraphs where sentences are not on separate lines. Note that `split('.')` *removes* the periods, which is why they are added back below.

```python
        embeddings = self.model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            if sim >= self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentences[i]]

        chunks.append(". ".join(current_chunk) + ".")
        return chunks
```
Identical algorithm to Cell 1. The only difference is the join: `". ".join(...) + "."` puts back the periods that `split('.')` removed, so the chunk reads as normal sentences again.

#### `split_documents()`: the LangChain-friendly interface

```python
    def split_documents(self,docs):
        result=[]
        for doc in docs:
            for chunk in self.split(doc.page_content):
                result.append(Document(page_content=chunk, metadata=doc.metadata))
        return result
```
- Takes a **list of `Document` objects** (what every LangChain loader returns) and returns a list of smaller `Document`s.
- For each input document, it splits the text with `self.split()`.
- Each chunk becomes a new `Document` that **inherits the original metadata**. This is important: if the original document had `{"source": "report.pdf", "page": 7}`, every chunk keeps that, so you can later cite where an answer came from.
- Using the same method name as LangChain's splitters (`split_documents`) means this class is a drop-in replacement for `RecursiveCharacterTextSplitter` in any pipeline.

---

### Cell 5: Wrap the sample text in a `Document`

```python
sample_text = """ ... same 5 sentences ... """
doc = Document(page_content=sample_text)
doc
```
Creates one `Document` with the sample text and empty metadata (`metadata={}`). In a real project this would come from a loader (`PyPDFLoader`, `TextLoader`, etc.).

---

### Cell 6: Run the custom chunker

```python
chunker=ThresholdSematicChunker(threshold=0.7)
chunks=chunker.split_documents([doc])
chunks
```
- Creates the chunker (loads the model).
- `split_documents([doc])` takes a **list**, so the single document is wrapped in `[ ]`.
- The result is the same 4 chunks as Cell 1, now as `Document` objects ready for a vector store:

```
[Document(page_content='LangChain is a framework for building applications with LLMs. Langchain provides modular abstractions to combine LLMs with tools like OpenAI and Pinecone.'),
 Document(page_content='You can create chains, agents, memory, and retrievers.'),
 Document(page_content='The Eiffel Tower is located in Paris.'),
 Document(page_content='France is a popular tourist destination.')]
```

---

### Cell 7: A simple RAG pipeline on the semantic chunks

```python
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("groq_api")
```
- `load_dotenv()` reads the `.env` file (in the current or a parent folder) and puts its `KEY=value` lines into the environment.
- The second line copies the key to the name `ChatGroq` expects.

#### Step 1: Embedding model

```python
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```
The **same** model used for chunking, now wrapped in LangChain's `Embeddings` interface so FAISS can call it. It runs locally and needs no API key. Using the same model for chunking, storing and querying keeps everything in one vector space.

#### Step 2: Vector store and retriever

```python
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()
```
- `FAISS.from_documents` embeds each of the 4 chunks and builds an in-memory FAISS index from the vectors, keeping the original text and metadata alongside.
- `as_retriever()` wraps the store in a **retriever**: an object with `.invoke(query)` that embeds the query and returns the most similar chunks. The default is the top **k = 4** results, so with only 4 chunks it returns all of them (ranked by similarity). Use `as_retriever(search_kwargs={"k": 2})` to return fewer.

#### Step 3: The LLM

```python
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ["GROQ_API_KEY"]
)
```
A client for **Groq**, a cloud service that runs open-weight models on very fast custom hardware (LPUs). `openai/gpt-oss-20b` is OpenAI's open-weight 20B model hosted on Groq. Groq has a free tier, which makes it handy for learning.

#### Step 4: Retrieve, then ask

```python
query = "Explain cosine similarity in embeddings"
docs = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in docs])

response = llm.invoke(f"Context:\n{context}\n\nQuestion: {query}")
print(response)
```

| Line | What it does |
|---|---|
| `retriever.invoke(query)` | Embeds the question and returns the closest chunks as a list of `Document`s. |
| `"\n\n".join([doc.page_content for doc in docs])` | Pulls the text out of each document and joins them with blank lines into one context string. |
| `llm.invoke(f"Context:...Question:...")` | Sends a hand-built prompt (context + question) to the model. |
| `print(response)` | Prints the full `AIMessage` object (`content='...'` plus metadata). Use `print(response.content)` to print only the answer text. |

**An important lesson hidden in this cell:** none of the 4 chunks mention cosine similarity. The retriever **always** returns its top-k results, even if they are irrelevant. The LLM then ignores the useless context and answers from its own training knowledge. That's why the output is a long, general explanation of cosine similarity. In a real system you would:
- tell the model in the prompt to answer **only** from the context and say "I don't know" otherwise, and/or
- use `search_type="similarity_score_threshold"` on the retriever to drop chunks below a minimum similarity.

---

### Cell 8: A proper prompt template

```python
template = """Answer the question based on the following context:

{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
```
- `from_template` scans the string for `{placeholders}` and detects `context` and `question` as the **input variables** (visible in the printed output).
- Unlike the f-string in Cell 7, a template is a reusable object that can be plugged into a chain and filled with different values each time.

---

### Cell 9: The same RAG pipeline as an LCEL chain

**LCEL (LangChain Expression Language)** lets you connect steps with the `|` (pipe) operator. The output of each step becomes the input of the next, like a Unix pipe.

```python
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.4
)
```
`temperature` controls randomness: `0` gives the most deterministic, focused answers, and higher values give more varied wording. `0.4` is a moderate setting suitable for factual Q&A.

```python
rag_chain = (
    RunnableMap(
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)
```
Data flow for `{"question": "What is LangChain used for?"}`:

```
{"question": "..."}
      │
      ▼
RunnableMap ── "context":  retriever.invoke(question) → list of chunks
            └─ "question": passed through unchanged
      │   {"context": [...], "question": "..."}
      ▼
prompt       fills {context} and {question} into the template
      ▼
llm          sends the filled prompt to Groq, gets an AIMessage back
      ▼
StrOutputParser  extracts .content → plain string
```

| Piece | Role |
|---|---|
| `RunnableMap({...})` | Runs both lambdas on the same input and builds a dict with keys matching the prompt's placeholders. |
| `lambda x: retriever.invoke(x["question"])` | Fetches relevant chunks for the question. |
| `lambda x: x["question"]` | Passes the question through so the prompt can use it too. |
| `prompt` | Turns the dict into the final prompt text. |
| `llm` | Generates the answer. |
| `StrOutputParser()` | Returns just the answer text, so `print(result)` shows clean text instead of a message object. |

```python
query = {"question": "What is LangChain used for?"}
result = rag_chain.invoke(query)
print(result)
```
This time the question **is** covered by the chunks, so the answer is grounded in them:

> LangChain is a framework that lets you build applications powered by large language models (LLMs). It offers modular abstractions for combining LLMs with external tools (such as OpenAI, Pinecone, etc.) and for creating components like chains, agents, memory modules, and retrievers.

Every fact in that answer comes from chunks 1 and 2. That is RAG working as intended.

> **Small improvement:** `retriever.invoke()` returns a *list of `Document` objects*, so the prompt receives their Python representation (`[Document(metadata={}, page_content='...'), ...]`). It works, but the clean way is to join the texts first:
> ```python
> "context": lambda x: "\n\n".join(d.page_content for d in retriever.invoke(x["question"])),
> ```

---

### Cell 10: Imports for LangChain's built-in chunker

```python
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
```
- `SemanticChunker` is LangChain's ready-made semantic splitter. It lives in `langchain_experimental`, which means its API may still change.
- `TextLoader` reads a `.txt` file into a `Document`.
- `OpenAIEmbeddings` is imported but not used. Any LangChain embeddings object works with `SemanticChunker`.

---

### Cell 11: `SemanticChunker` in action

```python
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunker = SemanticChunker(embedding)

loader = TextLoader("langchain_intro.txt")
docs = loader.load()

chunks = chunker.split_documents(docs)

print(f"Number of chunks: {len(chunks)}")
print(chunks[0].page_content)
```

| Line | What it does |
|---|---|
| `SemanticChunker(embedding)` | Creates the splitter with default settings (explained below). It needs only an embeddings object. |
| `TextLoader("langchain_intro.txt")` | Points at the sample file (same 5 sentences). The path is relative, so run the notebook from inside `Semantic-Chunking/`. |
| `loader.load()` | Returns a list with one `Document`, with `metadata={"source": "langchain_intro.txt"}`. |
| `chunker.split_documents(docs)` | Same interface as every LangChain splitter, and as our custom class. |

Output:
```
Number of chunks: 2
LangChain is a framework for building applications with LLMs. Langchain provides modular abstractions to combine LLMs with tools like OpenAI and Pinecone. You can create chains, agents, memory, and retrievers.
```
The second chunk is the two Paris/France sentences. **This is the ideal split:** one chunk per topic, which our fixed-threshold version at `0.7` did not achieve.

#### How `SemanticChunker` works (and why it did better)

It follows the same basic idea as our class, with three refinements:

1. **Smarter sentence splitting.** It splits on `.`, `?` or `!` followed by whitespace (regex `(?<=[.?!])\s+`), instead of on every period.

2. **Context buffer (`buffer_size=1`).** Before embedding, each sentence is combined with its neighbour on each side. So it embeds *"S1 S2"*, *"S1 S2 S3"*, *"S2 S3 S4"*, and so on. Short sentences are noisy on their own; adding neighbours gives a steadier signal.

3. **Relative threshold instead of a fixed one.** It computes the **distance** (`1 − cosine similarity`) between each pair of neighbouring windows, then splits only at the **largest** jumps. With the default `breakpoint_threshold_type="percentile"` and `breakpoint_threshold_amount=95`, it splits wherever the distance is above the 95th percentile of all distances in that document. Because the cut-off adapts to each document, you don't have to guess a magic number like `0.7` per model.

Other options for `breakpoint_threshold_type`:

| Type | Splits where the distance is… | Good for |
|---|---|---|
| `"percentile"` (default) | above the Xth percentile (default 95) | General use |
| `"standard_deviation"` | more than X standard deviations above the mean (default 3) | Documents with fairly uniform text |
| `"interquartile"` | above mean + X × IQR (default 1.5) | Robust to outliers |
| `"gradient"` | at sharp changes in the distance curve | Highly related text (legal, medical) where all similarities are high |

Useful extra parameters:
```python
SemanticChunker(
    embedding,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90,   # lower = more splits
    number_of_chunks=None,            # or ask for a target number of chunks
    min_chunk_size=None,              # merge tiny chunks
    buffer_size=1,                    # neighbours added on each side
)
```

---

## 5. Custom Chunker vs `SemanticChunker`

| | `ThresholdSematicChunker` (Cells 1, 4) | `SemanticChunker` (Cell 11) |
|---|---|---|
| Sentence splitting | Newlines (Cell 1) or every `.` (Cell 4) | `.`, `?`, `!` followed by whitespace |
| What is embedded | Each sentence alone | Each sentence plus neighbours (buffer) |
| Split rule | Fixed similarity threshold | Relative: largest distance jumps in the document |
| Tuning | Must pick a threshold per model / dataset | Works reasonably out of the box |
| Result on sample | 4 chunks at `0.7` | 2 chunks (one per topic) |
| Value | Shows exactly how the idea works | What to use in practice |

---

## 6. Limitations and Tips

- **Two embedding passes.** Every sentence is embedded during chunking, and each chunk is embedded again for the vector store. With a paid API this doubles cost; with a local model it doubles time.
- **Naive sentence splitting.** Splitting on `.` breaks on abbreviations (`e.g.`, `Dr.`, `U.S.`), decimals (`3.14`), URLs and file names (`readme.md`). For messy text, use a real sentence tokenizer such as `nltk.sent_tokenize` or spaCy.
- **No size limits.** Neither chunker caps chunk length. A long single-topic section can become one huge chunk that exceeds the embedding model's input limit (256 tokens for MiniLM, so the rest is ignored). A common fix is to semantically chunk first, then run `RecursiveCharacterTextSplitter` on any chunk that is too long.
- **Tiny chunks.** Single-sentence chunks (like chunk 2 in Cell 6) have little context and retrieve poorly. Lower the threshold, use `min_chunk_size`, or merge small chunks with their neighbour.
- **Threshold is model-specific.** A value that works for `all-MiniLM-L6-v2` will not transfer to `bge` or OpenAI embeddings, which have different similarity ranges.
- **Retrieval always returns something.** As Cell 7 showed, a retriever returns the top-k chunks even when none are relevant. Instruct the LLM to answer only from the context, or filter by a similarity score.
- **Keep metadata.** Always carry the source metadata into each chunk (as `split_documents` does) so answers can cite their source.

---

## 7. Setup

From the project root:

```bash
pip install -r requirements.txt
```

The packages this folder needs are `sentence-transformers`, `scikit-learn` (installed with sentence-transformers), `langchain-core`, `langchain-community`, `langchain-huggingface`, `langchain-experimental`, `langchain-groq`, `faiss-cpu` and `python-dotenv`.

Create a `.env` file in the project root with your Groq key (free at [console.groq.com](https://console.groq.com)):

```
groq_api=your_groq_key_here
```

Then open `code.ipynb` **from inside the `Semantic-Chunking/` folder** (so `langchain_intro.txt` is found) and run the cells in order. Cells 0, 1, 4–6, 10 and 11 need no API key, and Cells 3, 7 and 9 need the Groq key. Cell 11 uses `HuggingFaceEmbeddings`, which is imported in Cell 7, so run Cell 7 first (or add the import to Cell 10). The first run downloads the embedding model (about 90 MB); after that it works offline.

---

## 8. Quick Glossary

| Term | Meaning |
|---|---|
| **Chunk** | A small piece of a document that is embedded and stored on its own |
| **Fixed-size chunking** | Splitting by character or token count, ignoring meaning |
| **Semantic chunking** | Splitting where the meaning changes, using embedding similarity |
| **Breakpoint** | A position between two sentences where a split is made |
| **Threshold** | The similarity cut-off below which a new chunk starts |
| **Cosine similarity** | How closely two vectors point the same way; 1 = same meaning, ~0 = unrelated |
| **Cosine distance** | `1 − cosine similarity`; bigger = more different |
| **Buffer size** | How many neighbouring sentences are added to each sentence before embedding |
| **Percentile** | The value below which a given percent of data falls; the 95th percentile is the cut-off for the top 5% |
| **`Document`** | LangChain object holding `page_content` (text) and `metadata` (dict) |
| **FAISS** | Facebook AI Similarity Search, a fast in-memory vector index |
| **Retriever** | An object that takes a question and returns the most relevant chunks |
| **k** | How many chunks the retriever returns (default 4) |
| **LCEL** | LangChain Expression Language; connects steps with `\|` into a chain |
| **`RunnableMap`** | Runs several functions on one input and returns a dict of results |
| **`StrOutputParser`** | Turns the LLM's message object into a plain string |
| **Groq** | A cloud provider that serves open-weight LLMs very fast, with a free tier |
| **Temperature** | LLM randomness setting; low = focused and repeatable, high = varied |
