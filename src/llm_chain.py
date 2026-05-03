import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"  # or "mistral"


def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a financial assistant for a mutual fund company.

Answer the user's question using ONLY the context below.
Cite the FAQ question number in your answer (e.g., "Source: Q9").

If the answer is not in the context, say:
"I don't have enough information."

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


if __name__ == "__main__":
    from retriever import retrieve

    query = "If I redeem mutual funds after 14 months, how am I taxed?"

    results = retrieve(query)
    chunks = [r["text"] for r in results["results"]]

    answer = generate_answer(query, chunks)

    print("\nFinal Answer:\n")
    print(answer)