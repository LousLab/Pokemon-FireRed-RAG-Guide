"""
Entry point. Pipeline: load pre-chunked data -> embed -> index -> retrieve -> generate.
"""

import sys

from src.loader import load_chunks
from src.embedder import embed_texts
from src.vector_store import VectorStore
from src.retriever import retrieve
from src.generator import generate_answer

DATA_FOLDER = "data"
TOP_K = 3


def build_pipeline(data_folder: str = DATA_FOLDER) -> VectorStore:
    print("Loading chunks...")
    chunks = load_chunks(data_folder)
    if not chunks:
        print(f"No .jsonl files found in '{data_folder}/'.")
        sys.exit(1)

    print(f"Loaded {len(chunks)} chunk(s). Embedding...")
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)

    print("Building vector index...")
    store = VectorStore(embedding_dim=embeddings.shape[1])
    store.add(embeddings, chunks)

    return store


def run_chat_loop(store: VectorStore) -> None:
    print("\nReady. Ask about Pokémon FireRed (or 'quit' to exit).\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        retrieved = retrieve(query, store, top_k=TOP_K)

        print("\n--- Retrieved chunks ---")
        for chunk, score in retrieved:
            print(f"[{score:.3f}] {chunk['title']}")

        answer = generate_answer(query, retrieved)
        print("\n--- Answer ---")
        print(answer, "\n")


if __name__ == "__main__":
    vector_store = build_pipeline()
    run_chat_loop(vector_store)
