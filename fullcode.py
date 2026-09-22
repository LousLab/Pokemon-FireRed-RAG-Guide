import os
import sys
import json
import argparse
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads variables from a local .env file, if present
except ImportError:
    pass  # dotenv is optional; env vars can still be set manually

TOP_K = 3


# ---------- 1. LOAD ----------
def load_chunks(path: Path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ---------- 2. EMBED ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


# ---------- 3. INDEX ----------
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product == cosine similarity on normalized vectors
    index.add(embeddings)
    return index


# ---------- 4. RETRIEVE ----------
def retrieve(query, index, chunks, top_k=TOP_K):
    query_vec = embed_texts([query])
    scores, idxs = index.search(query_vec, top_k)
    return [(chunks[i], float(scores[0][rank])) for rank, i in enumerate(idxs[0])]


# ---------- 5. GENERATE ----------
def generate_answer(model, query, retrieved):
    context = "\n\n".join(f"- {chunk['content']}" for chunk, score in retrieved)

    prompt = f"""You are a helpful guide for Pokémon FireRed. Answer the
question using the context below. If the context doesn't contain the
answer, say so explicitly and then answer generally.

Context:
{context}

Question: {query}

Answer:"""

    response = model.generate_content(prompt)
    return response.text


# ---------- 6. MAIN LOOP ----------
def main():
    parser = argparse.ArgumentParser(description="RAG chatbot over a JSONL knowledge base.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("DATA_FILE", "data/pokemon_firered.jsonl")),
        help="Path to the JSONL knowledge base (default: data/pokemon_firered.jsonl, or $DATA_FILE)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "GEMINI_API_KEY is not set.\n"
            "Set it as an environment variable, or put it in a local .env file "
            "(see .env.example). Never commit real keys to the repo."
        )

    if not args.data.exists():
        sys.exit(f"Data file not found: {args.data}\nPass a valid path with --data or set DATA_FILE.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    print("Loading chunks...")
    chunks = load_chunks(args.data)
    print(f"Loaded {len(chunks)} chunk(s). Embedding...")

    embeddings = embed_texts([c["content"] for c in chunks])

    print("Building index...")
    index = build_index(embeddings)

    print("\nReady. Ask about Pokémon FireRed (or 'quit' to exit).\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        retrieved = retrieve(query, index, chunks)

        print("\n--- Retrieved chunks ---")
        for chunk, score in retrieved:
            print(f"[{score:.3f}] {chunk['title']}")

        answer = generate_answer(model, query, retrieved)
        print("\n--- Answer ---")
        print(answer, "\n")


if __name__ == "__main__":
    main()
