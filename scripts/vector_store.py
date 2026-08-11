import faiss
import numpy as np
import json
from pathlib import Path


def build_index(chunks: list[dict]) -> tuple:
    embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype="float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks


def search(query_embedding: list, index, chunks: list[dict], top_k: int = 3) -> list[dict]:
    query = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "score": float(distances[0][i]),
            "text": chunks[idx]["text"]
        })

    return results


if __name__ == "__main__":
    from ingest import load_document, chunk_text
    from embed import embed_chunks, model
    import sys

    file = sys.argv[1]
    query = sys.argv[2]

    text = load_document(file)
    chunks = chunk_text(text)
    chunks = embed_chunks(chunks)

    index, chunks = build_index(chunks)

    query_embedding = model.encode([query])[0].tolist()
    results = search(query_embedding, index, chunks)

    print(f"\nQuery: {query}\n")
    for r in results:
        print(f"Rank {r['rank']} (score: {r['score']:.4f})")
        print(r["text"])
        print("---")