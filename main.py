import sys
from src.transcriber import transcribe
from src.retriever import retrieve
from src.generator import generate_answer


AUDIO_PATH = "input/investor_sample.mp3"


def build_context(retrieved_docs):
    """
    Convert retrieved Chroma results into clean context string
    for LLM grounding.
    """
    context_parts = []

    for r in retrieved_docs["results"]:
        text = r["text"]
        source = r.get("source", "unknown")
        qnum = r.get("q_number", "unknown")

        context_parts.append(f"[{source} | Q{qnum}] {text}")

    return "\n\n".join(context_parts)


def main():
    print("\n=== Sunrise AMC Voice RAG Pipeline ===\n")

    # -----------------------
    # STEP 1: TRANSCRIPTION
    # -----------------------
    print("[1] Transcribing audio...")

    transcript_result = transcribe(AUDIO_PATH)

    query = transcript_result["transcript"]

    print("\nUser Query:")
    print(query)

    # -----------------------
    # STEP 2: RETRIEVAL
    # -----------------------
    print("\n[2] Retrieving relevant FAQs...")

    retrieved = retrieve(query, n_results=3)

    context = build_context(retrieved)

    # -----------------------
    # STEP 3: GENERATION
    # -----------------------
    print("\n[3] Generating answer using LLM (Ollama)...\n")

    answer = generate_answer(query, context)

    # -----------------------
    # OUTPUT
    # -----------------------
    print("\n=== FINAL ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()