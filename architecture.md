# Notes

## Why this data
Pokémon FireRed mechanics (types, gyms, battles) are simple, factual,
and easy to verify by eye — good for checking if retrieval is actually
pulling the right chunk, without needing domain expertise to judge it.

## Goal
- **Our goal:** learn how a RAG pipeline works end to end — chunking,
  embedding, vector search, retrieval, grounding an LLM's answer.
- **The model's goal:** act as a guide that answers Pokémon FireRed
  questions using the knowledge base, not guesses.

## Pipeline

1. **Load** — read the data (`.jsonl`, already chunked; or `.txt`, chunked by us).
2. **Embed** — turn each chunk into a vector (local, `sentence-transformers`).
3. **Index** — store vectors in FAISS for similarity search (local, in-memory).
4. **Retrieve** — embed the user's question, find the closest chunks.
5. **Generate** — send the question + retrieved chunks to Gemini, get an answer.

```
data → embed → FAISS index
                    ↑
              query embed → search → top-k chunks → Gemini → answer
```

## Setup
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
python main.py
```

## .gitignore
Keeps generated/local stuff out of the repo: virtual envs, `__pycache__`,
and API keys. Data files are kept in this case since they're the
point of the project (small, public knowledge base — nothing private).
