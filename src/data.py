# src/data.py
from pathlib import Path
from typing import List


def load_documents(data_dir: str) -> List[dict]:
    """Load all .txt documents from a directory."""
    documents = []
    for file in Path(data_dir).glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        if text.strip():  # Skip empty files
            documents.append({"id": file.stem, "text": text})
    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks of words."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))  # Prevent slice overshoot
        chunks.append(" ".join(words[start:end]))
        if end == len(words):  # Reached end, stop
            break
        start = end - overlap

    return chunks


def prepare_chunks(raw_documents: List[dict]) -> List[dict]:
    """Convert raw documents into labeled chunks."""
    documents = []
    for doc in raw_documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            documents.append({"id": f"{doc['id']}_{i}", "text": chunk})
    return documents
