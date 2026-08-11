from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest import load_document, chunk_text
from scripts.embed import embed_chunks, model
from scripts.vector_store import build_index, search
from scripts.llm import ask_llm

app = FastAPI(
    title="AI RAG Knowledge Assistant",
    version="0.2.0"
)

# Load and index documents at startup
text = load_document("data/Vasanth_resume_Developer.pdf")
chunks = chunk_text(text)
chunks = embed_chunks(chunks)
index, chunks = build_index(chunks)


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "AI RAG Knowledge Assistant"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask")
def ask(body: Question):
    query_embedding = model.encode([body.question])[0].tolist()
    results = search(query_embedding, index, chunks)
    answer = ask_llm(body.question, results)
    return {
        "question": body.question,
        "answer": answer,
        "sources": [r["text"][:200] for r in results]
    }