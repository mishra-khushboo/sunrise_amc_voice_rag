"""
ingestion.py
Ingests SunriseAMC_FAQ.pdf, chunks it intelligently,
embeds each chunk, and stores in ChromaDB.
"""

import re
import logging
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = "data/chroma_db"
COLLECTION = "sunrise_faq"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []

    for page in doc:
        text = page.get_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


# -----------------------------
# SMART FAQ CHUNKING
# -----------------------------
def split_by_faq_questions(text: str) -> list[dict]:
    """
    Smart chunking: detect FAQ question boundaries using regex.
    Falls back to sliding-window if not detected.
    """

    pattern = re.compile(
        r'((?:Q\.?|Question|FAQ)?\s*\d+[\.):]\s)',
        re.IGNORECASE
    )

    parts = pattern.split(text)
    chunks = []

    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        content = parts[i + 1].strip()

        combined = f"{header} {content}".strip()

        if combined:
            q_num_match = re.search(r'\d+', header)
            q_num = q_num_match.group() if q_num_match else str(len(chunks) + 1)

            chunks.append({
                "text": combined,
                "source": f"FAQ_Q{q_num}",
                "q_number": q_num,
            })

        i += 2

    # fallback if FAQ detection fails
    if len(chunks) < 3:
        logger.warning("FAQ pattern not detected — falling back to sliding window chunking.")
        chunks = sliding_window_chunks(text)

    logger.info(f"Created {len(chunks)} chunks")
    return chunks


# -----------------------------
# FALLBACK CHUNKING
# -----------------------------
def sliding_window_chunks(text: str) -> list[dict]:
    """Fallback: sliding window with overlap."""
    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": f"chunk_{idx}",
                "q_number": str(idx),
            })

        start = end - CHUNK_OVERLAP
        idx += 1

    return chunks


# -----------------------------
# MAIN INGESTION PIPELINE
# -----------------------------
def ingest(
    pdf_path: str = "input/SunriseAMC_FAQ.pdf",
    force_reingest: bool = False
) -> chromadb.Collection:
    """
    Full ingestion pipeline: extract → chunk → embed → store.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    existing = [c.name for c in client.list_collections()]

    # skip if already exists
    if COLLECTION in existing and not force_reingest:
        logger.info("Collection already exists — skipping ingestion.")
        return client.get_collection(COLLECTION)

    # delete if re-ingesting
    if COLLECTION in existing:
        logger.info("Deleting existing collection for re-ingestion...")
        client.delete_collection(COLLECTION)

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # STEP 1: extract
    logger.info(f"Extracting text from: {pdf_path}")
    text = extract_text_from_pdf(str(pdf_path))

    if not text.strip():
        raise ValueError("Extracted text is empty. Check your PDF.")

    # STEP 2: chunk
    chunks = split_by_faq_questions(text)

    # STEP 3: embed
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]

    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True
    ).tolist()

    # STEP 4: store
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f"{c['source']}_{i}" for i, c in enumerate(chunks)],
        metadatas=[
            {"source": c["source"], "q_number": c["q_number"]}
            for c in chunks
        ],
    )

    logger.info(f"Stored {len(chunks)} chunks in ChromaDB collection '{COLLECTION}'")

    return collection


# -----------------------------
# TEST RUN
# -----------------------------
if __name__ == "__main__":
    collection = ingest(force_reingest=True)

    # quick test query
    results = collection.query(
        query_texts=["tax on mutual fund redemption"],
        n_results=2
    )

    print("\nSample Query Result:")
    print(results)