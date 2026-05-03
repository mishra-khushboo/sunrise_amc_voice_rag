import requests
import time

def generate_answer(query, context):
    url = "http://localhost:11434/api/generate"

   
    context = context[:400]

    prompt = f"""
You are a financial assistant for Sunrise AMC.

Answer ONLY using the context below.

Context:
{context}

Question:
{query}

Rules:
- Be precise
- Mention FAQ number if present
- Do NOT hallucinate

Answer:
"""

    payload = {
    "model": "llama3",
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.2,
        "num_predict": 200
    }
}

    try:
        print("→ Sending request to Ollama...")

        start = time.time()

        response = requests.post(
            url,
            json=payload,
            timeout=300
        )

        print("→ Response received from Ollama")

        response.raise_for_status()

        result = response.json()["response"]

        print(f"→ LLM latency: {time.time() - start:.2f}s")

        return result

    except requests.exceptions.Timeout:
        return "Error: LLM request timed out (reduce context or warm model with 'ollama run llama3')"

    except Exception as e:
        return f"Error calling Ollama: {str(e)}"