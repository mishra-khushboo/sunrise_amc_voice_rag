

1. Model Selection

Speech-to-Text (Faster-Whisper)

* Model used: `faster-whisper (base)`
* Reason for selection:

  * Optimized for CPU inference, which aligns with the assignment constraint of running on a standard developer laptop.
  * Provides a strong balance between latency and transcription accuracy compared to larger Whisper variants.
  * Supports word-level timestamps and confidence scores, which are required for structured JSON output in Part 1.
* Tradeoff:

  * Lower accuracy compared to `small/medium` Whisper models, especially in noisy environments.
  * Chosen intentionally to ensure reproducible performance without GPU dependency.



Embedding Model (Sentence Transformers)

* Model used: `all-MiniLM-L6-v2`
* Reason for selection:

  * Lightweight (~80MB) and fast to compute embeddings.
  * Produces 384-dimensional embeddings suitable for semantic similarity tasks.
  * Well-tested for retrieval-augmented generation (RAG) pipelines and performs well on FAQ-style datasets.
* Tradeoff:

  * Less expressive than larger models (e.g., `bge-large` or `mpnet-base`), but significantly faster and memory-efficient.
  * Prioritized speed and local execution over marginal accuracy gains.



### LLM (Ollama - Llama 3)

* Model used: `llama3` via Ollama local runtime
* Reason for selection:

  * Fully open-source and compliant with assignment constraint (no paid APIs).
  * Strong instruction-following capability for grounded QA generation.
  * Easily runnable locally via `ollama serve`, eliminating external API dependency.
* Tradeoff:

  * High memory footprint (~4–5GB model weight) leading to cold-start latency.
  * Response time is hardware-dependent and can vary significantly on CPU-only machines.

---

## 2. Chunking Strategy

### Approach Used: Hybrid FAQ-aware + Sliding Window Fallback

The ingestion pipeline uses a two-stage chunking strategy:

#### (A) FAQ Structure Detection (Primary)

* Uses regex-based detection of patterns like:

  * `Q1.`, `Question 1`, `FAQ 1`, etc.
* Each detected question-answer pair is treated as a single semantic unit.
* This preserves the natural structure of financial FAQ documents.

#### (B) Sliding Window Fallback (Secondary)

* If FAQ structure detection fails or produces insufficient chunks:

  * A sliding window approach is used.
  * Chunk size: **400 characters**
  * Overlap: **80 characters**

### Reasoning:

* FAQ documents are inherently structured → preserving Q&A pairs improves retrieval accuracy.
* Sliding window ensures robustness for unstructured or poorly formatted sections.
* Overlap prevents semantic loss at chunk boundaries, which is critical for partial queries (e.g., taxation, redemption rules).

---

## 3. Tradeoffs Made

### 1. Latency vs Accuracy

* Chose lightweight models (MiniLM, Whisper base) instead of larger high-accuracy models.
* Result: faster execution on CPU, but slightly lower semantic precision and transcription accuracy.

---

### 2. Local Execution vs Performance

* Entire pipeline runs locally using:

  * Whisper (CPU)
  * ChromaDB (local vector store)
  * Ollama LLM (local inference)
* Tradeoff:

  * No dependency on cloud APIs ensures compliance and reproducibility.
  * However, LLM inference speed is significantly slower than hosted APIs.

---

### 3. Simplicity vs Optimized RAG Architecture

* Used basic cosine similarity search without:

  * reranking models
  * hybrid BM25 + vector retrieval
* This simplifies implementation but reduces retrieval precision in edge cases.

---

### 4. Character-based chunking vs Token-based chunking

* Used character-level chunking for simplicity.
* Tradeoff:

  * Easier implementation
  * Less precise alignment with LLM token context boundaries

---

## 4. Production Readiness Considerations

If this system were deployed in production, the following improvements would be necessary:

### 1. Performance & Scalability

* Replace local LLM (Ollama) with:

  * GPU-optimized inference (vLLM or TensorRT-LLM)
* Add request batching and caching layer for frequent queries
* Introduce async processing for transcription + LLM calls

---

### 2. Retrieval Quality Improvements

* Replace current embedding-only retrieval with:

  * Hybrid search (BM25 + vector similarity)
* Add reranking model (e.g., cross-encoder) to improve top-k relevance
* Implement query rewriting for better semantic matching

---

### 3. Chunking Improvements

* Replace character-based chunking with token-aware chunking (tiktoken / HF tokenizer-based)
* Add semantic chunking based on sentence boundaries instead of fixed size windows

---

### 4. Observability & Monitoring

* Add logging for:

  * retrieval accuracy
  * LLM latency
  * transcription confidence
* Track hallucination rate via evaluation dataset

---

### 5. Fault Tolerance

* Add retry logic for Ollama API failures
* Add fallback responses when LLM is unavailable
* Graceful degradation to retrieval-only answers if generation fails


