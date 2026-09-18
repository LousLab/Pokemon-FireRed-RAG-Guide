"""
Vector storage and similarity search.

FAISS is a local library, not a server or hosted database — everything
here runs in-process, in memory. IndexFlatIP does exact (brute-force)
search using inner product, equivalent to cosine similarity on
normalized vectors.

Each stored chunk is a dict (e.g. {"id", "title", "topic", "content"})
so metadata travels alongside the embedded text.
"""

import faiss
import numpy as np


class VectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: list[dict] = []  # keeps metadata aligned with index positions

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """Add embeddings and their corresponding chunk dicts to the store."""
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[tuple[dict, float]]:
        """
        Find the top_k chunks most similar to a query embedding.

        Returns:
            List of (chunk_dict, similarity_score) tuples, best match first.
        """
        scores, indices = self.index.search(query_embedding, top_k)
        results = [
            (self.chunks[i], float(scores[0][rank]))
            for rank, i in enumerate(indices[0])
        ]
        return results
