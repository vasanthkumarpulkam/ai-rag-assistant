from pathlib import Path
import pypdf


def read_pdf(file_path: Path) -> str:
    reader = pypdf.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def read_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def load_document(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix == ".pdf":
        return read_pdf(path)
    elif path.suffix == ".txt":
        return read_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            "index": index,
            "text": chunk,
            "start": start,
            "end": end
        })
        start += chunk_size - overlap
        index += 1

    return chunks


if __name__ == "__main__":
    import sys
    file = sys.argv[1]
    text = load_document(file)
    chunks = chunk_text(text)
    print(f"--- Loaded: {file} ---")
    print(f"Total characters: {len(text)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"\nChunk 0:\n{chunks[0]['text']}")
    print(f"\nChunk 1:\n{chunks[1]['text']}")