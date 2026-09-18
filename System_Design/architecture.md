# Architecture

The system has 5 components, each doing one job:

| Component | File | Job |
|---|---|---|
| Loader | `src/loader.py` | reads data off disk into memory |
| Embedder | `src/embedder.py` | turns text into vectors (local model) |
| Vector Store | `src/vector_store.py` | stores vectors, does similarity search (FAISS) |
| Retriever | `src/retriever.py` | embeds a query, asks the store for matches |
| Generator | `src/generator.py` | builds the prompt, calls Gemini |

Each one only talks to the next. `main.py` just wires them together in
order. Nothing else is coupled to anything else — you can replace the
generator with a different LLM, or the vector store with a different
library, without touching the other pieces.

## Why this data
Pokémon FireRed mechanics (types, gyms, battles) are simple and
factual — easy to check by eye whether retrieval pulled the *right*
chunk, without needing domain expertise.

## Goal
- **Our goal:** learn how a RAG pipeline works end to end.
- **The model's goal:** answer Pokémon FireRed questions using the
  knowledge base, not guesses.
