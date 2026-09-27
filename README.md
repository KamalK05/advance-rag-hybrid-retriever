# ⚡ Advance RAG with Hybrid Retrieval (BM25 + FAISS)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-VectorStore-0487D9?style=for-the-badge)
![BM25](https://img.shields.io/badge/BM25-KeywordSearch-orange?style=for-the-badge)

<p align="center">
  <b>Hybrid Retrieval System</b> combining dense vector search (FAISS) with sparse keyword matching (BM25) to maximize precision and recall across technical knowledge bases.
</p>

</div>

---

## 📌 Executive Summary

**Advance RAG Hybrid Retriever** solves a classic information retrieval dilemma: dense vector embeddings excel at capturing deep semantics but often miss exact keyword matches (like specific product names or error codes). By fusing **FAISS** (semantic search) with **BM25** (lexical keyword search), this repository ensures you never miss critical context when querying complex domain data.

---

## ⚡ Key Features

| Component | Description | Source Module |
| :--- | :--- | :--- |
| **🔍 FAISS Retriever** | Dense vector similarity search using OpenAI or Gemini embeddings. | `utils/retriever_faiss.py` |
| **🔤 BM25 Retriever** | Lexical sparse retrieval optimized for keyword matching and exact terms. | `utils/retriever_bm25.py` |
| **🔀 Hybrid Orchestrator** | Blends and re-ranks results from both vector and keyword search pipelines. | `utils/hybrid_retriever.py` |
| **📂 Document Pipeline** | Automated loaders and text chunkers for raw text corpora (`ai_docs/`). | `utils/loader.py` |
| **⚙️ Configurable Core** | Environment controls and central YAML configurations for dynamic tuning. | `config.yaml`, `.env` |

---

## 🏗️ Project Architecture

```
                               ┌───────────────────────────┐
                               │       User Query          │
                               └─────────────┬─────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
        ┌───────────────────────────┐                 ┌───────────────────────────┐
        │      FAISS Retriever      │                 │       BM25 Retriever      │
        │  (utils/retriever_faiss.py│                 │  (utils/retriever_bm25.py)│
        └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     Hybrid Retriever      │
                               │(utils/hybrid_retriever.py)│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     LLM Answer Gen        │
                               │        (app.py)           │
                               └───────────────────────────┘
```

---

## 📂 Project Structure

```
advance-rag-hybrid-retriever/
├── 📁 data/
│   ├── 📁 embeddings/
│   │   ├── 📁 faiss_openai/              # OpenAI vector index store
│   │   └── 📁 faiss_gemini/              # Gemini vector index store
│   └── 📁 raw/
│       └── 📁 ai_docs/
│           ├── 📄 langchain_guide.txt    # Sample framework documentation
│           └── 📄 transformers_in_ai.txt # Sample architecture guide
│
├── 📁 utils/                              # Core Processing Modules
│   ├── 📄 loader.py                       # Document ingestion pipeline
│   ├── 📄 retriever_faiss.py              # Dense vector search handler
│   ├── 📄 retriever_bm25.py               # Sparse keyword search handler
│   ├── 📄 hybrid_retriever.py             # Fusion and ranking logic
│   └── 📄 common_utils.py                 # Shared helper functions
│
├── 📄 app.py                              # Main execution entry point
├── 📄 config.yaml                         # System parameters & retrieval settings
├── 📄 template.py                         # Scaffold generator script
├── 📄 requirements.txt                    # Project dependencies
└── 📄 .env                                # Secure API keys configuration
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Environment Setup
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and supply your provider credentials:
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Review Configuration (`config.yaml`)
Tune your hybrid weight distribution, chunk sizes, and embedding models inside `config.yaml`:
```yaml
chunk_size: 1000
chunk_overlap: 200
top_k: 4
hybrid_weight_vector: 0.7
hybrid_weight_keyword: 0.3
```

---

## 💻 Running the Application

Execute the pipeline script to run ingestion and test retrieval:
```bash
python app.py
```

---

## 🤝 Contributing

1. Create your feature branch (`git checkout -b feature/HybridOptimization`)
2. Commit your changes (`git commit -m 'Add custom RRF (Reciprocal Rank Fusion) reranking'`)
3. Push to the branch (`git push origin feature/HybridOptimization`)
4. Open a Pull Request