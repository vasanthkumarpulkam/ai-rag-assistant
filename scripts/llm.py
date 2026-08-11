import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def ask_llm(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join([f"[Chunk {r['rank']}]:\n{r['text']}" for r in context_chunks])

    prompt = f"""You are a helpful assistant. Answer the question using only the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    response = httpx.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-oss-20b:free",
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30
    )

    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    from ingest import load_document, chunk_text
    from embed import embed_chunks, model
    from vector_store import build_index, search
    import sys

    file = sys.argv[1]
    question = sys.argv[2]

    text = load_document(file)
    chunks = chunk_text(text)
    chunks = embed_chunks(chunks)
    index, chunks = build_index(chunks)

    query_embedding = model.encode([question])[0].tolist()
    results = search(query_embedding, index, chunks)

    answer = ask_llm(question, results)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")