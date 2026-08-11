from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    return chunks


if __name__ == "__main__":
    from ingest import load_document, chunk_text
    import sys

    file = sys.argv[1]
    text = load_document(file)
    chunks = chunk_text(text)
    chunks = embed_chunks(chunks)

    print(f"Total chunks embedded: {len(chunks)}")
    print(f"Embedding size: {len(chunks[0]['embedding'])}")
    print(f"First 5 values of chunk 0 embedding: {chunks[0]['embedding'][:5]}")