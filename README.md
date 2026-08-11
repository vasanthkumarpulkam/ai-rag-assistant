# AI RAG Knowledge Assistant

A production-style Retrieval-Augmented Generation (RAG) pipeline built from scratch — no LangChain, no shortcuts.

## What it does

Upload any PDF or text document. Ask a question. Get an answer grounded in your document with sources cited.

## Architecture
Documents (PDF / TXT)
↓
Ingestion Pipeline
↓
Text Chunking (500 chars, 50 overlap)
↓
Embeddings (sentence-transformers / all-MiniLM-L6-v2)
↓
Vector Store (FAISS → OpenSearch)
↓
POST /ask {"question": "..."}
↓
Semantic Search → Top-K Chunks
↓
LLM (OpenRouter / Bedrock)
↓
{"answer": "...", "sources": [...]}
## Tech Stack

- **Python 3.12**
- **FastAPI** — REST API
- **sentence-transformers** — local embeddings (no API key needed)
- **FAISS** — local vector search
- **OpenRouter** — LLM API (swappable with AWS Bedrock)
- **Docker** — containerization (Phase 5)
- **AWS** — S3, OpenSearch, Bedrock (Phase 5)

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | FastAPI skeleton with `/` and `/health` endpoints | ✅ Done |
| 2 | Document ingestion — PDF and TXT support | ✅ Done |
| 3 | Chunking + embeddings + FAISS vector search | ✅ Done |
| 4 | LLM integration + `POST /ask` endpoint | ✅ Done |
| 5 | OpenSearch + Docker + AWS (S3, Bedrock) | 🔄 In progress |

## Local Setup

```bash
git clone https://github.com/vasanthkumarpulkam/ai-rag-assistant.git
cd ai-rag-assistant

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:
# OPENROUTER_API_KEY=your_key_here
Add a document to the `data/` folder, then run:

```bash
fastapi dev app/main.py
```

Open `http://127.0.0.1:8000/docs` to test the API.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service status |
| GET | `/health` | Health check |
| POST | `/ask` | Ask a question |

### Example

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the technical skills?"}'
```

```json
{
  "question": "What are the technical skills?",
  "answer": "...",
  "sources": ["...", "...", "..."]
}
```

## Author

Vasanth Kumar Pulkam — Data Engineer / AI Engineer
