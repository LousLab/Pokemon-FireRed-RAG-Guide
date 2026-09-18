# Pokémon FireRed RAG Guide

A small RAG (Retrieval-Augmented Generation) pipeline built from scratch,
no LangChain/LlamaIndex, using a Pokémon FireRed knowledge base as the
test data. Ask it questions about type matchups, gym order, mechanics,
etc., and it answers using retrieved facts instead of guessing.

Built to learn how RAG actually works under the hood: chunking,
embedding, vector search, retrieval, and grounding an LLM's answer in
retrieved context.

## Pipeline

```
data/*.jsonl (pre-chunked facts)
      │
      ▼
   Embed (sentence-transformers, local)
      │
      ▼
   FAISS vector index (local, in-memory)
      │
   query → embed → search → top-k chunks
      │
      ▼
   Gemini generates answer using retrieved chunks as context
```

## Project structure

```
├── main.py              # entry point
├── data/
│   └── pokemon_firered.jsonl   # knowledge base, one fact per line
├── requirements.txt
└── src/
    ├── loader.py          # reads data (.txt or pre-chunked .jsonl)
    ├── chunker.py           # splits raw .txt into chunks (unused for this dataset)
    ├── embedder.py            # text -> vector
    ├── vector_store.py         # FAISS similarity search
    ├── retriever.py              # embed query + search
    └── generator.py                # builds prompt, calls Gemini
```

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"   # free key: aistudio.google.com/apikey
```

## Usage

```bash
python main.py
```

Then ask things like:

```
You: What is Charizard weak to?
You: What order are the gym leaders in?
You: Is Rock super effective against Charizard?
```

## Notes

- Data is already chunked (one JSON object per fact), so `chunker.py`
  is unused here — it's there for plain `.txt` input if you swap datasets.
- `IndexFlatIP` is exact brute-force search, fine at this scale.
- Only the generation step calls an API; embedding and search are local.
