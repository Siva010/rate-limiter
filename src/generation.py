# src/generation.py
from groq import Groq
from config import LLM_MODEL, TEMPERATURE, GROQ_API_KEY

_client: Groq | None = None


def _get_client() -> Groq:
    """Lazily initialize and return the Groq client."""
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def build_prompt(context: str, query: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )


def generate_answer(prompt: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant. Answer the question using ONLY the provided context. If the answer is not in the context, say 'I don't know'."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()
