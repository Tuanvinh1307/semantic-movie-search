# MovieScout AI — Semantic Movie Search (Advanced RAG)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-red)
![Groq](https://img.shields.io/badge/Groq-Llama--3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)

**MovieScout AI** is a movie information retrieval system built on an **Advanced Retrieval-Augmented Generation (RAG)** architecture. Instead of traditional keyword matching, it lets users search for movies using **semantics, context, or vague fragments of memory** (e.g. *"a sci-fi space movie where a father leaves a watch for his daughter"*).

The project was built to solve the **vocabulary mismatch** problem in Information Retrieval (IR) — handling thousands of movie records while returning results in under 2 seconds.

---

## Key Features

The system follows an **Adaptive Cascade Retrieval** architecture, built on these core techniques:

* **Hybrid Search** — Combines the semantic understanding of **Dense Vectors** (`all-MiniLM-L6-v2`) with the keyword-matching strength of **Sparse Vectors** (Okapi BM25 via `fastembed`). Scores are fairly merged using **Reciprocal Rank Fusion (RRF)**.
* **Difficulty Router** — Automatically analyzes the **score gap** of first-pass results to branch the query. Easy queries return results immediately (early exit) for speed; hard queries are routed to deeper processing.
* **Query Expansion with HyDE & Guardrail** — Uses **Llama-3.1-8b** (via Groq Cloud) to generate a "hypothetical plot" (HyDE) for ambiguous queries. A dedicated **HyDE Validation module**, powered by a Cross-Encoder (threshold −2.0), acts as a hallucination guardrail and automatically discards off-topic LLM output.
* **Cross-Encoder Reranking** — Uses `ms-marco-MiniLM-L-6-v2` to score deep cross-attention between the query and each candidate passage, producing high-precision rankings (high MRR & NDCG).
* **Qdrant Cloud Payload Indexing** — Applies metadata pre-filtering directly at the vector-database level to filter by genre and release year with near-zero latency.

---

## System Architecture

The system is split into two main pipelines:

1. **Offline Pipeline (Data Ingestion & Embedding)**
   * Crawls high-quality data from the **TMDB API** (movies from 1990–2026, rating > 6.5).
   * Preprocesses and chunks text (recursive chunking with `tiktoken`).
   * Batch-encodes vectors and upserts them into **Qdrant Cloud** using UUID v5 identifiers to prevent duplicates.
2. **Online Pipeline (Adaptive Retrieval)**
   * Accepts a query from the **Streamlit** UI.
   * Processing flow: *Query Processing → 1st Hybrid Search → MaxP Aggregation → Difficulty Routing → HyDE (LLM) → HyDE Validation → 2nd Hybrid Search → Cross-Encoder Rerank → Final Scoring.*

---

## Installation & Setup

### 1. Requirements
* Python 3.10+
* Qdrant Cloud, Groq Cloud, and TMDB accounts (for API keys)

### 2. Install dependencies

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Tuanvinh1307/semantic-movie-search.git
cd semantic-movie-search
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file and fill in your own API keys:

```bash
cp .env.example .env
```

### 4. Run the app

```bash
streamlit run ui/app_final.py
```

---

## Evaluation

The `evaluation/` folder contains the retrieval benchmark used to validate the system, including a 200-query evaluation set and comparison results against a baseline keyword search, scored with standard IR metrics (MRR, NDCG).

---

## Tech Stack

`Python` · `Qdrant Cloud` · `Groq (Llama-3.1-8b)` · `Streamlit` · `Sentence-Transformers` · `fastembed (BM25)` · `Cross-Encoder Reranker`

---
