# Diagram

```
 data/*.jsonl
      │
      ▼
   Embed ──▶ FAISS index
                  ▲
                  │ search
                  │
   question ──▶ Embed ──▶ top-k chunks ──▶ Gemini ──▶ answer
```
