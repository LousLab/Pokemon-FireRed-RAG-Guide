# Pipeline

The order things happen in, every time you run `main.py`:

1. **Load** — read `data/*.jsonl` (already chunked) or `data/*.txt` (raw).
2. **Embed** — turn each chunk into a vector.
3. **Index** — store vectors in FAISS for fast similarity search.
4. **Retrieve** — embed the user's question, find the closest chunks.
5. **Generate** — send question + retrieved chunks to Gemini, get an answer.

Steps 1–3 happen once, at startup. Steps 4–5 happen every time you ask
a question.
