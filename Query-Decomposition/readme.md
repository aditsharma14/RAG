# Query Decomposition for RAG: Splitting a Complex Question into Sub-Questions

This folder covers **query decomposition**, a *query enhancement* technique: before retrieval, an LLM breaks a complex, multi-part question into 2–4 simple sub-questions. Each sub-question is retrieved and answered on its own, and a final LLM call combines the sub-answers into one answer.

A question like *"How does LangChain use memory and agents compared to CrewAI?"* asks about **three things at once** (LangChain memory, LangChain agents, CrewAI). One embedding of the whole question lands somewhere "in the middle" and tends to retrieve chunks for only one or two of those topics. Asking each part separately gives every topic its own search.

[Query-Expansion](../Query-Expansion/readme.md) makes **one** question longer; decomposition turns it into **several** smaller ones. Both work on the question, not on the chunks (compare [MMR](../MMR/readme.md) and [Reranking](../Reranking/readme.md)).

The notebook builds the pipeline in five steps:

1. **Load and chunk a text file**: `TextLoader` plus `RecursiveCharacterTextSplitter` turn `langchain_crewai_dataset.txt` into 241 chunks.
2. **Vector store and retriever**: FAISS with `all-MiniLM-L6-v2` embeddings, searched with MMR (`k=4`, `lambda_mult=0.7`).
3. **Decomposition chain**: `PromptTemplate | llm | StrOutputParser()` plus a small `decompose()` function turn a question into a Python list of sub-questions.
4. **Sub-question answer chain**: `create_stuff_documents_chain` answers each sub-question from its own retrieved chunks, or says *"Not in the documents."*
5. **Synthesis chain**: one more LLM call merges all sub-answers into a final answer, wrapped in a `decomposition_rag()` function.

```
Query-Decomposition/
├── code.ipynb                    # Notebook: decompose → retrieve + answer each sub-question → synthesize
├── langchain_crewai_dataset.txt  # 200 lines: 20 paragraphs about LangChain and CrewAI, each repeated 10 times (v1–v10)
└── readme.md                     # This file
```

---

## 1. Core Concepts

### The problem: one question, several topics
Vector search turns the whole question into **one** vector and looks for the chunks closest to it. When the question mixes several topics, that vector is an average of all of them, and the closest chunks are usually the ones that mention *several* of the words at once, not the ones that best explain each topic.

In this notebook the original question retrieves (Cell 5):

```
- CrewAI is compatible with LangChain agents and tools, allowing hybrid systems ...   ← mentions both frameworks
- LangChain agents operate using a planner-executor model ...
- CrewAI is compatible with LangChain agents and tools ...                            ← copy
- CrewAI is compatible with LangChain agents and tools ...                            ← copy
```

**No chunk about memory at all**, even though "memory" is in the question. The "CrewAI is compatible with LangChain agents" paragraph matches three of the question's words, so it wins three of the four slots.

### Query decomposition
**Definition:** An extra step before retrieval in which an LLM splits a complex question into simpler, self-contained sub-questions. Each sub-question is retrieved (and usually answered) separately, and the results are combined into one final answer.

```
"How does LangChain use memory and agents compared to CrewAI?"
                     │  decomposition_chain (LLM)
                     ▼
 ┌───────────────────────────────┬───────────────────────────────┬──────────────────────────────┐
 │ How does LangChain implement  │ What kinds of agents does     │ How does CrewAI differ from  │
 │ memory management?            │ LangChain support?            │ LangChain in memory/agents?  │
 └──────────────┬────────────────┴──────────────┬────────────────┴──────────────┬───────────────┘
                ▼                               ▼                               ▼
        retriever → qa_chain            retriever → qa_chain            retriever → qa_chain
                │                               │                               │
                └───────────────────────────────┼───────────────────────────────┘
                                                ▼
                                 synthesis_chain (LLM) → final answer
```

### Where it sits in the pipeline

```
question ──► decompose() ──► [subq1, subq2, subq3]
                                 │
                   for each subq │   retriever(subq) ──► chunks ──► qa_chain ──► "Q: subq\nA: answer"
                                 ▼
                        sub_answers (one string)
                                 │
question ───────────────────────►├──► synthesis_chain ──► final answer
```

The **original** question is used only twice: to create the sub-questions, and in the synthesis prompt so the final answer addresses what the user actually asked.

### Query enhancement family

| Technique | What the LLM produces | Retriever calls |
|---|---|---|
| Query expansion ([folder](../Query-Expansion/readme.md)) | One longer query with synonyms and related terms | 1 |
| Query rewriting | One cleaner, better-phrased query | 1 |
| Multi-query | Several *rephrasings* of the same question; results are merged | several |
| **Query decomposition** (this folder) | Several *different* sub-questions, one per topic | one per sub-question |
| HyDE | A fake answer paragraph, embedded instead of the question | 1 |

Multi-query and decomposition look similar, but **multi-query asks the same thing in different words**, while **decomposition asks different things**. Decomposition also answers each sub-question before combining, instead of just merging the chunks.

### Decomposition vs a single query

| | Single query | Decomposition |
|---|---|---|
| LLM calls per question | 1 | 1 (decompose) + N (sub-answers) + 1 (synthesis) = **5** for 3 sub-questions |
| Retriever calls | 1 | N |
| Multi-topic questions | One or two topics crowd out the rest | Each topic gets its own retrieval |
| Simple questions | Works well | Wasted calls; sub-questions can be near-duplicates |
| Traceability | One answer | Sub-questions and sub-answers show which part came from where |
| Risk | Missing a topic | A bad sub-question, or a "Not in the documents." that the synthesis papers over |

---

## 2. Use Cases

Query decomposition helps when the **question contains several questions**:

- **Comparisons**: *"How does X differ from Y?"* needs facts about X *and* about Y, which usually live in different chunks.
- **Multi-part questions**: *"What are the memory types, and how do agents use them?"* Each "and" is a candidate sub-question.
- **Multi-hop questions**: the answer to one part is needed to ask the next (*"Who maintains the library that CrewAI integrates with, and what license is it under?"*). The simple version in this notebook runs the sub-questions independently; a multi-hop version feeds earlier answers into later sub-questions.
- **Reports and summaries**: *"Give me an overview of X's retrieval, memory and agent features."* Each feature becomes one search.
- **Explainable answers**: the list of sub-questions and sub-answers shows the user (or you, while debugging) exactly what was looked up.

It is **less useful** when:
- The question is about one thing. *"What memory modules does LangChain have?"* gains nothing from being split.
- Latency or cost matters. A 3-sub-question run makes five LLM calls instead of one.
- The answer lives in one chunk that mentions everything. Splitting can make each sub-question miss that combined chunk.

---

## 3. Libraries Used

| Library | Purpose in this notebook |
|---|---|
| `langchain` | `init_chat_model`, which creates a chat model from a `"provider:model"` string |
| `langchain-groq` | `ChatGroq`, created behind the scenes by `init_chat_model("groq:...")` |
| `langchain-core` | `PromptTemplate`, `StrOutputParser` and the `\|` (pipe) operator |
| `langchain-community` | `TextLoader` to read the `.txt` file and the `FAISS` vector store wrapper |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` to cut the text into chunks |
| `langchain-huggingface` | `HuggingFaceEmbeddings`, the LangChain wrapper around sentence-transformers |
| `sentence-transformers` | Downloads and runs the `all-MiniLM-L6-v2` embedding model locally |
| `faiss-cpu` | The vector index that does the fast similarity / MMR search |
| `langchain-classic` | `create_stuff_documents_chain` (moved here from `langchain` in LangChain 1.x) |
| `python-dotenv` | Loads the Groq API key from a `.env` file |

---

## 4. The Code, Cell by Cell

### Cell 0: Title (Markdown)
A one-line summary of the idea: *split a complex question into simple sub-questions → retrieve and answer each → combine into one final answer.*

---

### Cell 1: Imports

```python
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
```

| Import | What it is |
|---|---|
| `init_chat_model` | Creates a chat model from a `"provider:model"` string. |
| `PromptTemplate` | A reusable prompt with `{placeholders}` filled in at run time. |
| `StrOutputParser` | Turns the LLM's message object into a plain Python string. |
| `TextLoader` | Reads a plain text file into a list with one `Document`. |
| `RecursiveCharacterTextSplitter` | Splits text into chunks of a maximum size, preferring to break at paragraphs, then lines, then spaces. |
| `HuggingFaceEmbeddings` | Turns text into vectors with a local sentence-transformers model. |
| `FAISS` | LangChain's wrapper around the FAISS index. |
| `create_stuff_documents_chain` | Builds a chain that "stuffs" a list of documents into one prompt variable and calls the LLM. |

The `DeprecationWarning` that `langchain-community` is being sunset is only a notice; everything still works. In the future `TextLoader` and `FAISS` will move to standalone packages.

---

### Cell 2: Load the API key and the LLM

```python
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("groq_api")

llm = init_chat_model("groq:openai/gpt-oss-20b")
llm.invoke("Say hi in 3 words").content
```

- `load_dotenv()` reads the `.env` file and puts its `KEY=value` lines into the environment.
- The next line copies the key from the `.env` name (`groq_api`) to the name `ChatGroq` looks for (`GROQ_API_KEY`). If `groq_api` is missing, `os.getenv` returns `None` and this line raises a `TypeError`.
- `init_chat_model("groq:openai/gpt-oss-20b")`: `groq` picks the provider, the rest is the model name (OpenAI's open-weight 20B reasoning model hosted on Groq).
- `llm.invoke("Say hi in 3 words")` is a quick test that the key works. Output: `'Hello there friend.'`

The same `llm` is used for **all three** jobs in this notebook: decomposing, answering sub-questions, and synthesizing.

---

### Cell 3: Load and chunk the text file

```python
loader = TextLoader("langchain_crewai_dataset.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(len(chunks))
```

| Line | What it does |
|---|---|
| `encoding="utf-8"` | The file contains a curly apostrophe (`LangChain’s`). Without it, Windows may fail to read the file. The path is relative, so run the notebook from inside `Query-Decomposition/`. |
| `chunk_size=300` | Each chunk is at most 300 characters. |
| `chunk_overlap=50` | When a paragraph has to be cut, up to 50 characters are repeated in the next chunk. |

Output: **241 chunks**.

#### About the dataset
The file is the same one used in [Query-Expansion](../Query-Expansion/readme.md): **200 lines**, but only **20 different paragraphs** (10 about LangChain, 10 about CrewAI), each repeated **10 times** and tagged `(v1)` to `(v10)`. Two facts about it matter for this notebook:

- LangChain memory has its own paragraph (*"LangChain offers memory modules like ConversationBufferMemory and ConversationSummaryMemory..."*).
- **CrewAI memory has no paragraph.** The only mention is one word in the YAML paragraph: *"specifying agent roles, goals, **memory**, and tools"*. This explains the *"Not in the documents."* answers in Cells 8 and 11.

---

### Cell 4: Vector store and retriever

```python
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "lambda_mult": 0.7})
```

| Line | What it does |
|---|---|
| `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` | Loads the small, free embedding model (384-number vectors, runs on CPU). The first run downloads about 90 MB. |
| `FAISS.from_documents(...)` | Embeds all 241 chunks and builds an in-memory index. |
| `search_type="mmr"` | Uses [Maximal Marginal Relevance](../MMR/readme.md) to try to avoid near-duplicates. |
| `"k": 4` | Return 4 chunks per search. Fewer than in Query-Expansion (5), because each sub-question is narrow and gets its own search. |
| `"lambda_mult": 0.7` | Leans towards **relevance** (1.0 = pure relevance, 0.0 = pure diversity). The default is 0.5. |

The `Loading weights` progress bar and the `unauthenticated requests to the HF Hub` warning are normal.

---

### Cell 5: Retrieval with the original question (baseline)

```python
query = "How does LangChain use memory and agents compared to CrewAI?"
for d in retriever.invoke(query):
    print("-", d.page_content[:100])
```

Output:

| # | Chunk |
|---|---|
| 1 | CrewAI is compatible with LangChain agents and tools, allowing hybrid systems… |
| 2 | LangChain agents operate using a planner-executor model… |
| 3 | CrewAI is compatible with LangChain agents and tools… (copy) |
| 4 | CrewAI is compatible with LangChain agents and tools… (copy) |

This is the problem decomposition solves: the question asks about memory, agents and CrewAI, but the four chunks cover only **agents** and the **LangChain/CrewAI link**. The memory paragraph is not retrieved at all. With `lambda_mult=0.7` and a dataset where every paragraph has 10 copies, MMR still returns three copies of the same paragraph.

---

### Cell 6: The decomposition chain

```python
decomposition_prompt = PromptTemplate.from_template("""
Break the question below into 2 to 4 simple sub-questions. Each sub-question must cover one topic only.
Return ONLY the sub-questions, one per line. No numbering, no bullets, no explanations, no intro text.

Question: {question}
""")
decomposition_chain = decomposition_prompt | llm | StrOutputParser()
```

| Part | Why it is there |
|---|---|
| *"2 to 4 simple sub-questions"* | A limit keeps the number of LLM and retriever calls small. Without it the model may produce 6–8 overlapping questions. |
| *"Each sub-question must cover one topic only"* | The whole point: one topic per search. |
| *"one per line. No numbering, no bullets..."* | Makes the output easy to turn into a Python list with `splitlines()`. |
| `prompt \| llm \| StrOutputParser()` | LCEL: fill the prompt, call the LLM, return the reply as a plain string. |

---

### Cell 7: Turn the LLM output into a list

```python
def decompose(question):
    text = decomposition_chain.invoke({"question": question})
    return [line.strip("-•*0123456789. ").strip()
            for line in text.splitlines()
            if line.strip().endswith("?")]

sub_questions = decompose(query)
sub_questions
```

| Part | What it does |
|---|---|
| `text.splitlines()` | One line per sub-question. |
| `if line.strip().endswith("?")` | Keeps only real questions. Blank lines and intro text like *"Here are the sub-questions:"* are dropped, even if the model ignores the "no intro text" instruction. |
| `line.strip("-•*0123456789. ")` | Removes any leftover bullets or numbering (`- `, `• `, `1. `) from **both ends** of the line. The trailing `?` stops it at the end. |
| `.strip()` | Removes the remaining whitespace. |

Output:

```python
['How does LangChain implement memory in its agent framework?',
 'How does CrewAI handle memory for its agents?',
 "What are the key differences between LangChain's and CrewAI's approaches to memory and agents?"]
```

Three sub-questions, but notice they are not perfectly "one topic only": the first mixes memory and agents, and the third is still a comparison.

---

### Cell 8: The sub-question answer chain

```python
qa_prompt = PromptTemplate.from_template("""
Answer using ONLY the context. If the context doesn't contain the answer, say "Not in the documents."

Context:
{context}

Question: {input}
""")
qa_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

# test on the first sub-question
subq = sub_questions[0]
qa_chain.invoke({"input": subq, "context": retriever.invoke(subq)})
```

| Part | Why it is there |
|---|---|
| *"Answer using ONLY the context"* | Each sub-answer must come from the documents, not from what the LLM already knows. |
| *"say 'Not in the documents.'"* | A fixed phrase the synthesis step (and you) can recognize when retrieval came up empty. |
| `{context}` | Filled by `create_stuff_documents_chain` with the retrieved chunks. |
| `{input}` | The sub-question. |

Unlike a `create_retrieval_chain`, retrieval is done **by hand** here (`retriever.invoke(subq)`) and passed in as `context`, so the same chain can be reused in the loop of Cell 10.

Output: `'Not in the documents.'`

Why? The first sub-question, *"How does LangChain implement memory in its agent framework?"*, retrieves **only agent chunks** (the planner-executor paragraph three times plus the "standout feature" agents paragraph). The words "agent framework" pulled the search towards agents, and the memory paragraph was not found. The planner-executor paragraph does say *"context-aware memory use across steps"*, but the LLM, told to use only the context, decided that's not an explanation of how memory is implemented. A sub-question that still mixes two topics has the same problem as the original question.

---

### Cell 9: The synthesis chain

```python
synthesis_prompt = PromptTemplate.from_template("""
Using the sub-answers below, write one clear answer to the original question.
Don't add facts that aren't in the sub-answers.

Original question: {question}

Sub-answers:
{sub_answers}
""")
synthesis_chain = synthesis_prompt | llm | StrOutputParser()
```

| Part | Why it is there |
|---|---|
| `{question}` | The **original** question, so the final answer is shaped around what the user asked, not around the sub-questions. |
| `{sub_answers}` | All `Q: ... A: ...` pairs in one block of text. |
| *"Don't add facts that aren't in the sub-answers."* | The synthesis LLM never sees the chunks, only the sub-answers. This line stops it from filling gaps with outside knowledge. |

---

### Cell 10: The full pipeline

```python
def decomposition_rag(question):
    sub_questions = decompose(question)
    pairs = []
    for subq in sub_questions:
        docs = retriever.invoke(subq)
        answer = qa_chain.invoke({"input": subq, "context": docs})
        pairs.append(f"Q: {subq}\nA: {answer}")
    sub_answers = "\n\n".join(pairs)
    final = synthesis_chain.invoke({"question": question, "sub_answers": sub_answers})
    return sub_questions, sub_answers, final
```

```
question
   │
   ▼ decompose()                         ← LLM call 1
[subq1, subq2, subq3]
   │
   ├─ subq1 → retriever → qa_chain       ← LLM call 2
   ├─ subq2 → retriever → qa_chain       ← LLM call 3
   └─ subq3 → retriever → qa_chain       ← LLM call 4
   │
   ▼ "Q: subq1\nA: ...\n\nQ: subq2\nA: ..."
synthesis_chain(question, sub_answers)  ← LLM call 5
   │
   ▼
(sub_questions, sub_answers, final)
```

- A plain Python `for` loop instead of an LCEL chain: easier to read and debug. The sub-questions run **one after another**; they could run in parallel (e.g. with `qa_chain.batch(...)`) since they don't depend on each other.
- It returns all three values, not just the final answer, so you can see which sub-question failed.
- `decompose()` is called **again** here, so the sub-questions are usually different from the ones printed in Cell 7 (LLM output varies between calls).

---

### Cell 11: Run it

```python
subs, sub_answers, final = decomposition_rag(query)
print("🧩 Sub-questions:", *subs, sep="\n- ")
print("\n📄 Sub-answers:\n", sub_answers)
print("\n✅ Final answer:\n", final)
```

This run produced **different** sub-questions from Cell 7:

| Sub-question | Retrieved (k=4) | Sub-answer |
|---|---|---|
| How does LangChain implement memory management within its framework? | memory modules ×2, modular workflows, LangChain intro | ✅ ConversationBufferMemory and ConversationSummaryMemory, to keep track of turns or summarize within token limits |
| What kinds of agents does LangChain support and how are they used? | "standout feature" agents ×3, planner-executor | ✅ Agents choose tools, plan and execute multi-step tasks, use branching and context-aware memory |
| How does CrewAI differ from LangChain in memory handling and agent usage? | "CrewAI is compatible with LangChain" ×3, CrewAI agents with purpose/goal/tools | ❌ Not in the documents. |

Final answer: a correct section on **LangChain memory** and a correct section on **LangChain agents**, followed by *"CrewAI differs from LangChain in memory handling and agent usage, but no specific details are given in the sub-answers."*

What this shows:
- **Decomposition worked for the LangChain parts.** The memory paragraph, which the original question never retrieved (Cell 5), is now found and used. The final answer is much richer than a single retrieval could give.
- **The CrewAI part failed**, for two reasons:
  1. The third sub-question is still a **comparison** that mixes CrewAI, LangChain, memory and agents, so it retrieves the "compatible with LangChain" paragraph again instead of CrewAI-specific chunks.
  2. The dataset has **no paragraph about CrewAI memory** (see Cell 3). A cleaner sub-question like *"How does CrewAI handle memory?"* (Cell 7) retrieves CrewAI agent chunks but still not the YAML paragraph with its single mention of memory.
- **The synthesis was honest.** Because of *"Don't add facts that aren't in the sub-answers"*, it said the details are missing instead of inventing a comparison. Its wording ("the provided information notes that CrewAI differs...") is a little misleading, though: the documents never said that; it came from the sub-question.

---

## 5. With vs Without Decomposition on This Data

Same question, same retriever (MMR, `k=4`, `lambda_mult=0.7`):

| | Single query (Cell 5) | Decomposition (Cell 11) |
|---|---|---|
| Retriever calls | 1 | 3 |
| Chunks seen | 4 | 12 |
| Distinct paragraphs | 2 (CrewAI–LangChain compatibility, planner-executor agents) | 7 (memory modules, modular workflows, LangChain intro, "standout feature" agents, planner-executor, CrewAI–LangChain compatibility, CrewAI agents) |
| LangChain memory covered? | ❌ | ✅ |
| LangChain agents covered? | ✅ | ✅ |
| CrewAI memory / agents covered? | Only the compatibility sentence | Only the compatibility sentence (answered "Not in the documents.") |
| LLM calls | 1 (to answer) | 5 |

Decomposition finds **more of the relevant topics**, at about five times the LLM cost. It does not create information that isn't in the documents; it just stops one topic from hiding the others.

---

## 6. Limitations and Tips

- **Sub-questions are only as good as the LLM makes them.** "One topic only" is a request, not a rule: in both runs the model produced a comparison sub-question that mixed topics, and that is the one that failed. Show it an example in the prompt: *"'How does X compare to Y?' → 'What is X?', 'What is Y?'"*, and leave the comparing to the synthesis step.
- **Comparison is the synthesis step's job.** Sub-questions should collect facts about each side separately; the final LLM call is where the two sides are compared.
- **The printed sub-questions are not the ones used.** Cell 7 and Cell 10 both call `decompose()`, and the LLM gave different lists. Generate them once and pass the list in, or use the `subs` that `decomposition_rag` returns.
- **Every question costs N + 2 LLM calls.** Use a small, fast model for decomposition, run the sub-answers in parallel (`qa_chain.batch`), or skip decomposition when the question has only one topic (e.g. ask the LLM to return the question unchanged if it is already simple).
- **"Not in the documents." can be lost in the synthesis.** Here it was handled well, but the synthesis LLM can also smooth over a gap. Check the sub-answers, or list unanswered sub-questions separately in the output.
- **Answering sub-questions isn't always needed.** A lighter version retrieves chunks for every sub-question, merges and removes duplicates, then answers the original question once. Fewer LLM calls; the final LLM sees all the chunks instead of summaries.
- **`decompose()` drops anything that doesn't end with `?`.** If the model writes a sub-question as an instruction (*"Explain LangChain memory."*), it is silently thrown away; if nothing ends with `?`, the list is empty and the loop answers nothing. Fall back to the original question when the list is empty.
- **The dataset is 10× duplicated.** Even with `lambda_mult=0.7`, most of the 4 slots per sub-question are copies. Removing duplicates before indexing would give each sub-question up to 4 *different* paragraphs.
- **Combine with other techniques.** Each sub-question can be [expanded](../Query-Expansion/readme.md) before retrieval, and each sub-question's chunks can be [reranked](../Reranking/readme.md) before answering.

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

Then open `code.ipynb` **from inside the `Query-Decomposition/` folder** (so `langchain_crewai_dataset.txt` is found) and run the cells in order. Cell 2 needs the API key right away (it runs a test call), and Cells 7, 8 and 11 call Groq. Cells 3–5 work without it. The first run downloads the embedding model (about 90 MB).

---

## 8. Quick Glossary

| Term | Meaning |
|---|---|
| **Query decomposition** | Using an LLM to split a complex question into simpler sub-questions that are retrieved and answered separately |
| **Sub-question** | One small, single-topic question produced from the original question |
| **Sub-answer** | The answer to one sub-question, from its own retrieved chunks |
| **Synthesis** | The final LLM step that combines the sub-answers into one answer to the original question |
| **Multi-hop question** | A question whose answer needs several facts, where one fact may be needed to find the next |
| **Query enhancement** | The family of techniques that improve the question before retrieval (expansion, rewriting, multi-query, decomposition, HyDE) |
| **MMR (Maximal Marginal Relevance)** | Retrieval that balances relevance to the query against similarity to chunks already picked |
| **`lambda_mult`** | MMR setting from 0 (pure diversity) to 1 (pure relevance); 0.7 here |
| **LCEL** | LangChain Expression Language: joining steps with `\|`, e.g. `prompt \| llm \| parser` |
| **`StrOutputParser`** | Turns an LLM message into a plain string |
| **Stuff chain** | A chain that puts all retrieved chunks into one prompt |
| **`init_chat_model`** | LangChain helper that creates a chat model from a `"provider:model"` string |
| **FAISS** | Facebook AI Similarity Search, a fast in-memory vector index |
| **Groq** | A cloud provider that serves open-weight LLMs very fast, with a free tier |
