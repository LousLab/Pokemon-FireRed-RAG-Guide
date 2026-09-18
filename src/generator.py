"""
Answer generation.

The only part of this pipeline that makes a network call. Retrieved
chunks are inserted into a prompt as grounding context, and Gemini is
asked to answer using that context.
"""

import os
import google.generativeai as genai

_MODEL_NAME = "gemini-2.0-flash"


def _get_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(_MODEL_NAME)


def build_prompt(query: str, retrieved_chunks: list[tuple[dict, float]]) -> str:
    """Assemble the retrieved context and user question into a single prompt."""
    context = "\n\n".join(f"- {chunk['content']}" for chunk, _score in retrieved_chunks)

    return f"""You are a helpful guide for Pokémon FireRed. Answer the
question using the context below. If the context doesn't contain the
answer, say so explicitly and then answer generally.

Context:
{context}

Question: {query}

Answer:"""


def generate_answer(query: str, retrieved_chunks: list[tuple[dict, float]]) -> str:
    """Call Gemini with the constructed prompt and return its text answer."""
    model = _get_model()
    prompt = build_prompt(query, retrieved_chunks)
    response = model.generate_content(prompt)
    return response.text
