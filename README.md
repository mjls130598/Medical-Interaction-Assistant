# 🏥 Medical Interaction Assistant

> **AI-powered chatbot for drug interaction and medication queries, grounded in official AEMPS pharmaceutical leaflets.**

![Status](https://img.shields.io/badge/status-in%20progress-yellow?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
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
│   │  (PyMuPDF)   │    │  (Splitter)   │    │ (OpenAI)   │   │
│   └──────────────┘    └───────────────┘    └──────┬─────┘   │
│                                                   │         │
│   ┌──────────────────────────────────────────┐    │         │
│   │         ChromaDB Vector Store            │◀───┘         │
│   │   (persistent embeddings of leaflets)    │              │
│   └───────────────────┬──────────────────────┘              │
│                       │  Similarity Search                  │
│                       ▼                                     │
│   ┌──────────────────────────────────────────┐              │
│   │         LLM (GPT via LangChain)          │              │
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
├── app/                        # Application source code
│   ├── core/                   # Core application logic and prompts
│   ├── rag/                    # RAG pipeline helpers and loaders
│   ├── __init__.py
│   └── main.py                 # FastAPI application entry point
│
├── data/                       # Input data and PDF storage
├── tests/                      # Pytest test suite
│   ├── integration/            # Integration tests
│   ├── unit/                   # Unit tests
│   └── conftest.py             # Pytest fixtures and setup
│
├── Pipfile                     # Dependency management (pipenv)
├── Pipfile.lock
├── pytest.ini                  # Pytest configuration
├── .gitignore
└── LICENSE                     # GPL-3.0
```

---

## ✅ Completed Steps

- [x] **PDF ingestion** — Load and parse official AEMPS drug leaflets using PyMuPDF
- [x] **Document chunking** — Split leaflets into semantically meaningful passages
- [x] **Test infrastructure** — Pytest + pyfakefs for filesystem-independent unit tests

---

## 🚧 Roadmap — Next Steps

- [ ] **Project scaffolding** — FastAPI app structure, routing, config management
- [ ] **Vector store setup** — ChromaDB integration with persistent storage of embeddings
- [ ] **RAG chain** — LangChain retrieval chain connecting ChromaDB ↔ OpenAI LLM
- [ ] **REST API layer** — FastAPI endpoints to receive queries and return answers
- [ ] **Expand the leaflet corpus** — Automate bulk ingestion from the AEMPS public API to cover the full catalogue of authorised medications
- [ ] **Multilingual support** — Extend queries to English (leaflets are in Spanish; add translation layer or multilingual embeddings)
- [ ] **Source citation in responses** — Surface the exact leaflet section used to ground each answer, with drug name and section reference
- [ ] **Drug interaction cross-queries** — Handle multi-drug questions (*"Can I take ibuprofen and omeprazole together?"*) by retrieving and combining context from multiple leaflets
- [ ] **Conversation memory** — Add LangChain `ConversationBufferMemory` to support follow-up questions within a session
- [ ] **Confidence scoring** — Flag responses where retrieved context similarity is below a threshold, adding a disclaimer for low-confidence answers
- [ ] **Streamlit / web UI** — Simple frontend for non-technical users to interact with the chatbot
- [ ] **CI/CD pipeline** — GitHub Actions workflow for automated testing on every push
- [ ] **Docker containerisation** — `Dockerfile` + `docker-compose` for reproducible local setup and deployment

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| API Framework | FastAPI |
| AI Orchestration | LangChain + LangChain-OpenAI |
| LLM | OpenAI GPT (via LangChain) |
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
- An OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/mjls130598/Medical-Interaction-Assistant.git
cd Medical-Interaction-Assistant

# Install dependencies
pipenv install

# Activate the virtual environment
pipenv shell

# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here
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
