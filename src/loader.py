"""
Document loading.

Supports two input formats:
- .txt files: raw, unchunked text (gets split later by chunker.py)
- .jsonl files: one JSON object per line, already chunked. Each object
  must have a "content" field (the chunk text). Any other fields
  (id, title, topic, etc.) are kept as metadata alongside the chunk.
"""

import os
import glob
import json


def load_documents(folder: str) -> list[str]:
    """Load every .txt file in `folder` as raw, unchunked text."""
    filepaths = glob.glob(os.path.join(folder, "*.txt"))
    documents = []
    for path in filepaths:
        with open(path, "r", encoding="utf-8") as f:
            documents.append(f.read())
    return documents


def load_chunks(folder: str) -> list[dict]:
    """
    Load every .jsonl file in `folder` as pre-made chunks.

    Each line must be a JSON object with at least a "content" field.
    Returns a list of dicts, e.g.:
        {"id": "pokemon_001", "topic": "...", "title": "...", "content": "..."}
    """
    filepaths = glob.glob(os.path.join(folder, "*.jsonl"))
    chunks = []
    for path in filepaths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks
