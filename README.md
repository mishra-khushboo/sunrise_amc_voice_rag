

# Sunrise AMC Voice-Powered Investor Support Assistant (RAG System)

## Overview

This project is a **Voice-Powered Retrieval-Augmented Generation (RAG) system** built for Sunrise Asset Management Co. Ltd.

It processes investor voice queries, converts speech to text, retrieves relevant information from a financial FAQ knowledge base, and generates grounded responses using a locally hosted LLM.

The system is fully **offline and open-source compliant**, using no paid APIs.

---

## System Architecture

The pipeline consists of four main stages:

### 1. Voice Transcription

* Input: `investor_sample.mp3`
* Engine: Faster-Whisper (base model)
* Output: Structured JSON with:

  * transcript text
  * word-level timestamps
  * confidence scores

---

### 2. Knowledge Base Ingestion

* Input: `SunriseAMC_FAQ.pdf`
* Processing:

  * PDF text extraction using PyMuPDF
  * Hybrid chunking strategy:

    * FAQ-aware regex chunking (primary)
    * Sliding window fallback (400 chars, 80 overlap)
* Storage:

  * ChromaDB persistent vector store

---

### 3. Embedding & Retrieval (RAG Layer)

* Embedding model: `all-MiniLM-L6-v2`
* Vector database: ChromaDB (cosine similarity)
* Retrieval:

  * Top-k similarity search
  * Metadata includes FAQ question numbers for citation

---

### 4. Answer Generation (LLM Layer)

* Model: `Llama 3` via Ollama (local inference)
* Input:

  * User query (from transcription)
  * Retrieved FAQ context
* Output:

  * Grounded answer based only on retrieved context
  * Includes FAQ-based citation when available

---

## Project Structure

```
sunrise_amc_voice_rag/
│
├── main.py                     # End-to-end pipeline
├── requirements.txt
├── README.md
├── DECISIONS.md
│
├── src/
│   ├── transcriber.py         # Whisper-based speech-to-text
│   ├── ingestion.py           # PDF ingestion + embedding pipeline
│   ├── retriever.py           # ChromaDB retrieval logic
│   ├── generator.py           # Ollama LLM integration
│
├── input/
│   ├── SunriseAMC_FAQ.pdf
│   ├── investor_sample.mp3
│
├── output/
│   ├── transcript.json
│
├── data/
│   └── chroma_db/             # Local vector database (ignored in git)
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Install Ollama (Local LLM)

```bash
ollama pull llama3
```

Start Ollama service:

```bash
ollama serve
```

---

### 3. Run the Full Pipeline

```bash
python main.py
```

---

## Output

The system produces:

### 1. Transcription Output

* Located in: `output/transcript.json`
* Includes:

  * full transcript
  * word timestamps
  * confidence scores

---

### 2. Retrieved Context

* Top matching FAQ sections
* Includes:

  * question number (e.g., Q9, Q10)
  * source metadata

---

### 3. Final Answer

* Generated via Llama 3 (Ollama)
* Grounded strictly in retrieved FAQ context
* Includes citation where applicable

---

## Design Highlights

### 1. Fully Offline Architecture

* No external APIs used
* Entire system runs locally (Whisper + ChromaDB + Ollama)

---

### 2. Hybrid Chunking Strategy

* Combines:

  * FAQ-aware semantic grouping
  * Sliding window fallback
* Ensures robustness across structured and unstructured PDFs

---

### 3. Lightweight Model Choices

* Whisper base → CPU-friendly transcription
* MiniLM embeddings → fast semantic search
* Llama 3 → strong instruction-following for local inference

---

## Limitations

* LLM latency depends on local hardware (cold start issue with Ollama)
* Character-based chunking may not be optimal for long financial documents
* No reranking model used (simplified retrieval pipeline)
* CPU-only execution leads to slower inference compared to GPU setups

---

## Future Improvements

* Introduce token-based semantic chunking
* Add reranker (cross-encoder) for improved retrieval accuracy
* Implement caching layer for repeated queries
* Optimize LLM inference using vLLM or GPU acceleration
* Add evaluation harness for retrieval + answer correctness scoring

---

## Compliance Notes

* ✔ Fully open-source stack
* ✔ No paid APIs used
* ✔ Runs locally on standard developer machine
* ✔ ChromaDB and model files excluded from repository

---

## Author Notes

This system is designed as a **production-inspired prototype**, prioritizing:

* reproducibility
* offline execution
* modular architecture
* explainability of design decisions


