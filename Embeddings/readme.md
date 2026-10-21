# Embeddings for RAG — Vectors, Cosine Similarity and HuggingFace Sentence Transformers

This folder covers **embeddings**, the stage of a RAG pipeline that turns text into numbers a computer can compare. Every earlier folder in this repository ends with clean, chunked `Document` objects; this folder shows what happens to them next. The notebook builds the idea up in three steps:

1. **Toy word embeddings in 2D**: hand-made vectors plotted on a chart, so you can *see* that similar words sit close together.
2. **Cosine similarity from scratch**: a four-line NumPy function that measures how alike two vectors are.
3. **Real embeddings with LangChain**: `HuggingFaceEmbeddings` and the free, local `all-MiniLM-L6-v2` model turn real sentences into 384-dimensional vectors, followed by a comparison of popular open-source embedding models.

```
Embeddings/
├── code.ipynb    # Notebook with all the experiments
└── readme.md     # This file
```

---

## 1. Core Concepts

### RAG (Retrieval-Augmented Generation)
**Definition:** A technique where a Large Language Model (LLM) answers questions using information *retrieved* from your own documents, rather than relying only on what it learned during training.

**Explanation:** A RAG pipeline usually has these stages:

```
Files ──► Parse ──► Chunk ──► Embed ──► Vector DB ──► Retrieve ──► LLM answer
                             └ this folder ┘
```

Embeddings are used **twice**: once offline, to turn every chunk into a vector before it is stored, and once per query, to turn the user's question into a vector so it can be compared against the stored ones. Retrieval is simply "find the stored vectors closest to the question vector".

### Embedding
**Definition:** A list of numbers (a **vector**) that represents the *meaning* of a piece of text. An embedding model is trained so that texts with similar meanings get vectors that point in similar directions.

**Explanation:** Computers cannot compare meanings directly, but they can compare numbers very fast. An embedding translates text into a "language" of numbers where:

- `"cat"` and `"kitten"` → vectors close together
- `"cat"` and `"car"` → vectors far apart, even though the words differ by only one letter

That last point is why embeddings beat keyword search: they match on **meaning**, not spelling. A question like *"How do I reset my password?"* can retrieve a chunk that says *"Steps to recover account credentials"* even though they share almost no words.

### Dimensions
**Definition:** The number of values in each vector. Each value (dimension) captures some learned aspect of meaning.

| Where | Dimensions | Notes |
|---|---|---|
| Toy example in the notebook | 2 (and 3) | Small enough to plot and reason about by hand |
| `all-MiniLM-L6-v2` | 384 | Fast, small, good general quality |
| `all-mpnet-base-v2` | 768 | Higher quality, slower |
| OpenAI `text-embedding-3-small` | 1536 | Hosted, needs an API key |

Individual dimensions in a real model have no human-readable meaning ("dimension 17 = animal-ness" is not how it works). Only the **overall direction** of the vector matters. More dimensions can capture finer distinctions, but cost more memory and slower search.

### Cosine Similarity
**Definition:** A measure of how similar two vectors are, based on the **angle** between them, ignoring their length.

```
                    A · B             (dot product)
cos_sim(A, B) = ─────────────  =  ─────────────────────────
                 ‖A‖ × ‖B‖         (length of A × length of B)
```

| Value | Angle | Meaning |
|---|---|---|
| **1** | 0° | Same direction: same meaning |
| **0** | 90° | Unrelated |
| **−1** | 180° | Opposite direction |

**Why angle and not distance?** A long document and a short sentence on the same topic can produce vectors of different lengths but similar directions. Cosine similarity ignores length, so it compares *topic*, not size. In practice, real sentence-embedding scores mostly fall between about 0 and 1; strongly negative scores like the toy example's are rare.

> **Tip:** Many models (including the sentence-transformers family) can return **normalized** vectors (length 1). For normalized vectors, cosine similarity equals the plain dot product, which is why vector databases often use "inner product" search.

### `embed_query` vs `embed_documents`
LangChain's embedding classes expose two methods:

| Method | Input | Output | Used for |
|---|---|---|---|
| `embed_query(text)` | one string | one vector (`list[float]`) | The user's question at query time |
| `embed_documents(texts)` | a list of strings | a list of vectors | The chunks at indexing time (batched, faster) |

For most sentence-transformers models both methods produce the same kind of vector. Some models (e.g. E5, BGE with instructions) add a different prefix to queries and documents, which is why LangChain keeps the two methods separate.

---

## 2. Libraries Used

| Library | Purpose in this notebook |
|---|---|
| `numpy` | Dot product (`np.dot`) and vector length (`np.linalg.norm`) for cosine similarity |
| `matplotlib` | Scatter plot of the 2D toy embeddings |
| `langchain-huggingface` | `HuggingFaceEmbeddings`, LangChain's wrapper around local HuggingFace models |
| `sentence-transformers` | The library that actually downloads and runs the model (used under the hood by `HuggingFaceEmbeddings`) |

---

## 3. The Code, Cell by Cell

### Cell 1: Imports

```python
import numpy as np
import matplotlib.pyplot as plt
```

### Cell 2: Toy word embeddings

```python
word_embeddings = {
    "cat": [0.8, 0.6],
    "kitten": [0.75, 0.65],
    "dog": [0.7, 0.3],
    "puppy": [0.65, 0.35],
    "car": [-0.5, 0.2],
    "truck": [-0.45, 0.15]
}
```

These vectors are **made up by hand**, not produced by a model. They are chosen to show the key property of embeddings: animals cluster on the right of the plane, vehicles on the left, and each young animal sits right next to its adult form.

### Cell 3: Plot the embeddings in 2D

```python
fig, ax = plt.subplots(figsize=(8, 6))
for word, coords in word_embeddings.items():
    ax.scatter(coords[0], coords[1], s=100)
    ax.annotate(word, (coords[0], coords[1]), xytext=(5, 5), textcoords='offset points')
...
plt.show()
```

Draws each word as a dot labelled with its name. The chart shows two clear groups: `cat / kitten / dog / puppy` in the upper right and `car / truck` on the left. **Distance on the chart ≈ difference in meaning.** Real models do the same thing in hundreds of dimensions, which we cannot draw, but the idea is identical.

### Cell 4: Cosine similarity from scratch

```python
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)
```

A direct translation of the formula in Section 1:

- `np.dot(vec1, vec2)` multiplies matching values and adds them up.
- `np.linalg.norm(v)` is the vector's length, `√(v₁² + v₂² + …)`.
- Dividing by both lengths removes the effect of size, leaving only the angle.

### Cell 5: Similar words

```python
cat_vector = [0.8, 0.6, 0.3]
kitten_vector = [0.75, 0.65, 0.35]
car_vector = [-0.5, 0.2, 0.1]

cosine_similarity(cat_vector, kitten_vector)
# → 0.9966
```

Note the vectors here are **3D** (a third value was added), showing the function works for any number of dimensions. `cat` vs `kitten` scores **≈ 0.997**: almost the same direction, so almost the same meaning.

### Cell 6: Unrelated words

```python
cosine_similarity(cat_vector, car_vector)
# → -0.4372
```

`cat` vs `car` scores **≈ −0.44**: the vectors point in clearly different directions. Same spelling distance as "cat → kitten" would suggest nothing; the embedding captures meaning instead.

For reference, the same function applied to the 2D vectors from Cell 2 gives:

| Pair | Cosine similarity |
|---|---|
| cat – kitten | 0.998 |
| car – truck | 0.998 |
| dog – puppy | 0.996 |
| cat – dog | 0.972 |
| cat – car | −0.520 |
| dog – truck | −0.747 |

### Cell 7: Load a real embedding model

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Downloads `all-MiniLM-L6-v2` (about 90 MB) from the HuggingFace Hub on first run and caches it locally; later runs load it from disk. It runs on your CPU, is free, and needs no API key.

> **Note:** The warning *"You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN…"* is harmless. It only means downloads are rate-limited. Set `HF_TOKEN` in your environment (a free token from huggingface.co) to remove it.

### Cell 8: Embed a single sentence

```python
text = "Hello, I am learning about embeddings!"
embedding = embeddings.embed_query(text)
print(len(embedding))   # → 384
print(embedding)        # → [-0.0181, -0.0995, 0.0138, ...]
```

The whole sentence becomes **one vector of 384 numbers**, no matter how long or short the sentence is. That fixed size is what makes it possible to compare any two texts.

### Cell 9: Embed several sentences

```python
sentences = [
    "The cat sat on the mat",
    "The cat sat on the mat",
    "The dog played in the yard",
    "I love programming in Python",
    "Python is my favorite programming language"
]
embedding_sentence = embeddings.embed_documents(sentences)
print(embedding_sentence[0])
print(embedding_sentence[1])
```

`embed_documents` embeds the whole list in one batched call and returns five 384-dimensional vectors. Sentences 0 and 1 are **identical**, so their printed vectors are identical too (cosine similarity = 1.0): embedding models are deterministic.

**Try it yourself:** compare every pair with the `cosine_similarity` function from Cell 4. You should see the two Python sentences score much higher with each other than with the cat or dog sentences, even though they share only the words "Python" and "programming".

```python
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity(embedding_sentence[i], embedding_sentence[j])
        print(f"{sim:.3f}  {sentences[i]!r}  vs  {sentences[j]!r}")
```

### Cell 10: Popular open-source embedding models

The final cell prints a comparison table of sentence-transformers models:

| Model | Dimensions | Strength | Best for |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fast and efficient, good quality | General purpose, real-time applications |
| `all-mpnet-base-v2` | 768 | Best quality, slower than MiniLM | When quality matters more than speed |
| `all-MiniLM-L12-v2` | 384 | Slightly better than L6, a bit slower | Good balance of speed and quality |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Optimized for question answering | Q&A systems, semantic search |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Supports 50+ languages | Multilingual applications |

Switching models is a one-line change to `model_name`. `time` is imported in this cell but not used yet; it is there for timing each model's speed.

---

## 4. Choosing an Embedding Model

- **Start with `all-MiniLM-L6-v2`.** It is small, fast, free and good enough for learning and many real projects.
- **Need better quality?** Try `all-mpnet-base-v2`, or a hosted model such as OpenAI's `text-embedding-3-small`.
- **Users ask questions, documents contain answers?** A Q&A-tuned model like `multi-qa-MiniLM-L6-cos-v1` matches question-style queries to answer-style text better.
- **Non-English text?** Use a multilingual model.
- **Check the model's maximum input length.** MiniLM models read up to 256 word pieces and silently cut off the rest, so chunks longer than that are only partly embedded. This is one reason chunk size matters.

> **Golden rule:** use the **same model** for documents and queries. Vectors from different models live in different "spaces" and cannot be compared. If you change the model, re-embed the whole collection.

---

## 5. Setup

From the project root:

```bash
pip install -r requirements.txt
```

The packages this folder needs are `numpy`, `matplotlib`, `langchain-huggingface` and `sentence-transformers` (which installs PyTorch). Then open `code.ipynb` and run the cells in order. The first run of Cell 7 needs an internet connection to download the model; after that it works offline.

---

## 6. Quick Glossary

| Term | Meaning |
|---|---|
| **Embedding** | A vector of numbers that represents the meaning of a text |
| **Vector** | An ordered list of numbers, e.g. `[0.8, 0.6]` |
| **Dimension** | One position in a vector; the vector's length is the model's dimension count (e.g. 384) |
| **Dot product** | Sum of element-wise products of two vectors: `a₁b₁ + a₂b₂ + …` |
| **Norm** | The length of a vector, `√(a₁² + a₂² + …)` |
| **Cosine similarity** | Dot product divided by both norms; 1 = same direction, 0 = unrelated, −1 = opposite |
| **Normalized vector** | A vector scaled to length 1; cosine similarity then equals the dot product |
| **Semantic search** | Finding text by meaning (vector similarity) rather than by matching keywords |
| **Sentence Transformers** | A family of open-source models (and a library) that embed whole sentences |
| **`HuggingFaceEmbeddings`** | LangChain class that runs a HuggingFace / sentence-transformers model locally |
| **`embed_query`** | Embed one string (the user's question) |
| **`embed_documents`** | Embed a list of strings (the chunks) in one batch |
