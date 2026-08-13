# AI RAG Knowledge Assistant

**A retrieval-augmented generation pipeline built from primitives — no LangChain, no LlamaIndex, no framework abstractions.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-vector_search-0467DF)](https://faiss.ai)
[![Sentence Transformers](https://img.shields.io/badge/sentence--transformers-local_embeddings-FFB000)](https://sbert.net)

---

## Overview

Point it at a PDF or text file. Ask a question. Get an answer grounded in that document, with the source passages returned alongside it.

Every stage is written out explicitly — ingestion, chunking, embedding, indexing, retrieval, prompt construction. The goal is to understand RAG rather than to call a framework that hides it.

## Architecture

```
Document (PDF / TXT)
        │
        ▼
  ingest.py       load_document()  →  pypdf or plain text
        │         chunk_text()     →  500 chars, 50-char overlap
        ▼
  embed.py        sentence-transformers / all-MiniLM-L6-v2
        │         384-dimensional vectors, computed locally — no API key
        ▼
  vector_store.py FAISS IndexFlatL2
        │
        ▼
  POST /ask  {"question": "..."}
        │
        ├── encode the question
        ├── search top-k = 3 nearest chunks
        │
        ▼
  llm.py          context + question → OpenRouter chat completion
        │         grounded prompt: answer only from context,
        │         otherwise say so explicitly
        ▼
  {"question": "...", "answer": "...", "sources": [...]}
```

## Features

- **PDF and plain-text ingestion** via `pypdf`
- **Overlapping chunking** — 500 characters with 50-character overlap so context isn't cut mid-sentence
- **Local embeddings** — `all-MiniLM-L6-v2` runs on your machine; no embedding API key needed
- **FAISS vector search** — exact L2 nearest-neighbour retrieval
- **Grounded prompting** — the model is instructed to answer only from retrieved context and to say "I don't have enough information" rather than guess
- **Sources returned** with every answer
- **Every module runs standalone** — each script has a `__main__` block for testing that stage in isolation

## Tech stack

| Component | Technology |
|---|---|
| API | FastAPI |
| PDF parsing | pypdf |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | FAISS (`IndexFlatL2`) |
| LLM | OpenRouter (swappable for AWS Bedrock, OpenAI, or a local model) |
| HTTP client | httpx |

## Getting started

### Prerequisites

- Python 3.12
- An [OpenRouter](https://openrouter.ai) API key (the default model is free-tier)

### Install

```bash
git clone https://github.com/vasanthkumarpulkam/ai-rag-assistant.git
cd ai-rag-assistant

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configure

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_key_here
```

### Add a document

```bash
mkdir -p data
cp /path/to/your-document.pdf data/
```

Then set the document path in `app/main.py`.

### Run

```bash
fastapi dev app/main.py
```

Open <http://127.0.0.1:8000/docs> for the interactive API explorer.

## Usage

### Ask a question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the core technical skills listed?"}'
```

```json
{
  "question": "What are the core technical skills listed?",
  "answer": "Python, SQL, FastAPI, AWS ...",
  "sources": [
    "...first 200 characters of the top-ranked chunk...",
    "...second...",
    "...third..."
  ]
}
```

### Run any stage on its own

```bash
python scripts/ingest.py       data/doc.pdf
python scripts/embed.py        data/doc.pdf
python scripts/vector_store.py data/doc.pdf "your query"
python scripts/llm.py          data/doc.pdf "your question"
```

Each prints its intermediate output — chunk counts, embedding dimensions, retrieval scores — which makes the pipeline easy to reason about.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service status |
| `GET` | `/health` | Health check |
| `POST` | `/ask` | Ask a question against the indexed document |

## Project structure

```
ai-rag-assistant/
├── app/
│   └── main.py            FastAPI app; builds the index at startup
├── scripts/
│   ├── ingest.py          Document loading and chunking
│   ├── embed.py           Embedding generation
│   ├── vector_store.py    FAISS index construction and search
│   └── llm.py             Prompt assembly and LLM call
├── tests/
├── requirements.txt
└── .env                   (gitignored)
```

## Design notes

**Why no framework?** LangChain and LlamaIndex are useful in production, but they hide exactly the parts worth understanding — how chunk size interacts with retrieval quality, what actually goes into the prompt, why a bad chunk boundary produces a confidently wrong answer. Writing it out makes the tradeoffs visible.

**Why `IndexFlatL2`?** Exact search. At document scale it's fast enough, and it removes approximate-search recall as a variable when debugging retrieval quality. `IndexIVFFlat` or HNSW would be the swap for larger corpora.

**Why 500/50 chunking?** Large enough to hold a complete thought, small enough that three chunks fit comfortably in the context window. The overlap prevents a sentence spanning a boundary from being lost by both chunks.

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | FastAPI skeleton with `/` and `/health` | ✅ Done |
| 2 | Document ingestion — PDF and TXT | ✅ Done |
| 3 | Chunking, embeddings, FAISS search | ✅ Done |
| 4 | LLM integration and `POST /ask` | ✅ Done |
| 5 | OpenSearch backend, Docker, AWS (S3, Bedrock) | 🔄 In progress |
| 6 | Document upload endpoint, index persistence, evaluation harness | 📋 Planned |

## Known limitations

- The document path is hardcoded in `app/main.py`; there's no upload endpoint yet
- The index is rebuilt in memory on every startup and is not persisted
- No chunk re-ranking — top-3 by L2 distance only
- `tests/` is scaffolded but empty



## Author

**Vasanth Kumar Pulkam** — Data / AI Engineer · [GitHub](https://github.com/vasanthkumarpulkam)
