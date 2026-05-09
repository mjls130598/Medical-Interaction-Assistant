# 🏥 Medical Interaction Assistant

> **AI-powered chatbot for drug interaction and medication queries, grounded in official AEMPS pharmaceutical leaflets.**

![Status](https://img.shields.io/badge/status-in%20progress-yellow?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-teal?style=flat-square&logo=fastapi)
![License](https://img.shields.io/badge/license-GPL--3.0-orange?style=flat-square)

---

## 💡 What is this?

Most people have asked questions like:

- *"Can a 5-year-old take ibuprofen?"*
- *"What foods should a person taking Sintrom avoid?"*
- *"Can I take paracetamol if I'm on antibiotics?"*

Getting reliable answers to these questions usually means digging through dense pharmaceutical leaflets — or worse, guessing.

**Medical Interaction Assistant** solves this by combining the official drug leaflets published by the **AEMPS** (Spanish Agency for Medicines and Health Products) with a **RAG (Retrieval-Augmented Generation)** pipeline, enabling natural-language queries answered with verified, source-grounded information.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
│         "Can a 5-year-old take ibuprofen?"                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI REST Layer                       │
│              POST /query  ·  GET /health                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangChain RAG Pipeline                     │
│                                                             │
│   ┌──────────────┐    ┌───────────────┐    ┌────────────┐   │
│   │  PDF Loader  │───▶│  Text Chunks  │───▶│ Embeddings │   │
│   │  (PyMuPDF)   │    │  (Splitter)   │    │ (GroQ)   │   │
│   └──────────────┘    └───────────────┘    └──────┬─────┘   │
│                                                   │         │
│   ┌──────────────────────────────────────────┐    │         │
│   │         ChromaDB Vector Store            │◀───┘         │
│   │   (persistent embeddings of leaflets)    │              │
│   └───────────────────┬──────────────────────┘              │
│                       │  Similarity Search                  │
│                       ▼                                     │
│   ┌──────────────────────────────────────────┐              │
│   │         LLM (GroQ via LangChain)         │              │
│   │   Prompt + retrieved context → Answer    │              │
│   └──────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Grounded, source-cited response                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Medical-Interaction-Assistant/
│
├── .github/                    # GitHub Actions CI workflows
│   └── workflows/
├── app/                        # Application source code
│   ├── config/                 # Configuration and logging helpers
│   │   ├── __init__.py
│   │   └── config_log.py
│   ├── core/                   # Core application logic and prompts
│   ├── rag/                    # RAG pipeline helpers and loaders
│   │   ├── cleaners/
│   │   ├── embedding/
│   │   ├── loaders/
│   │   └── readers/
│   ├── service/                # App service layer and orchestration
│   ├── __init__.py
│   └── main.py                 # FastAPI application entry point
│
├── data/                       # Input data and PDF storage
│   └── input_pdfs/
├── tests/                      # Pytest test suite
│   ├── integration/            # Integration tests
│   │   └── data/
│   └── unit/                   # Unit tests
│       ├── tests_app/
│       │   ├── tests_config/
│       │   ├── tests_rag/
│       │   │   ├── tests_cleaners/
│       │   │   ├── tests_loaders/
│       │   │   └── tests_readers/
│       │   └── tests_service/
│       │       ├── tests_vector_store/
│       │       ├── tests_assistance/
│       └── __init__.py
├── Pipfile                     # Dependency management (pipenv)
├── Pipfile.lock
├── pytest.ini                  # Pytest configuration
├── .gitignore
└── LICENSE                     # GPL-3.0
```

---

## ✅ Completed Steps

- [x] **PDF ingestion** — Load and parse official AEMPS drug leaflets using PyMuPDF
- [x] **Text cleaning** — Implement text cleaner components for document preprocessing
- [x] **Prospect loader** — Create a loader for pharmaceutical prospectus documents
- [x] **PDF document reader** — Build a PDF reader for structured leaflet extraction
- [x] **Vector database** — Create and configure a vector database for retrieval
- [x] **RAG system** — Build the retrieval-augmented generation communication system
- [x] **Source citation** — Insert consulted sources into each response
- [x] **CI workflows** — Add GitHub Actions workflows for tests, linting, security scan, and coverage

---

## 🚧 Roadmap — Next Steps

- [ ] **Create the FastAPI application**
- [ ] **Create a simple frontend for queries**
- [ ] **Create conversation memory for responses**

---

## 🔮 Future Implementations

- [ ] **Research document ingestion** — Add support for importing and indexing academic research papers
- [ ] **Multi-format ingestion** — Add support for web pages, DOCX files, and other document formats
- [ ] **Configure multi-language question handling** — Allow the system to accept and answer queries in multiple languages

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| API Framework | FastAPI |
| AI Orchestration | LangChain + GroQ |
| LLM | GroQ (via LangChain) |
| Vector Store | ChromaDB |
| PDF Processing | PyMuPDF |
| Testing | Pytest + pyfakefs |
| Dependency Management | Pipenv |
| Data Source | AEMPS (Agencia Española de Medicamentos y Productos Sanitarios) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [Pipenv](https://pipenv.pypa.io/en/latest/)
- A GroQ API key

### Installation

```bash
# Clone the repository
git clone https://github.com/mjls130598/Medical-Interaction-Assistant.git
cd Medical-Interaction-Assistant

# Install dependencies
pipenv install

# Activate the virtual environment
pipenv shell

# Set your GroQ API key
export GROQ_API_KEY=your_api_key_here
```

### Run the API

```bash
uvicorn app.main:app --reload
```

### Run the tests

```bash
pytest tests/
```

---

## ⚠️ Disclaimer

This tool is intended as an **informational aid only**. It does not replace professional medical advice. Always consult a qualified healthcare professional before making any medication decision. Responses are grounded in official AEMPS leaflets but may not reflect the most recent updates.

---

## 📄 License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

---

## 👩‍💻 Author

**María Jesús López Salmerón**
Python Developer · AI & Security  
[LinkedIn](https://linkedin.com/in/maria-jesus-lopez-salmeron) · [GitHub](https://github.com/mjls130598)
