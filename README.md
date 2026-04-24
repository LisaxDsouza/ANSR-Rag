# ANSR Knowledge Assistant (RAG)

A high-performance, cost-efficient Retrieval-Augmented Generation (RAG) tool designed for technical document analysis. This assistant supports multi-modal ingestion (PDF, Word, Excel, Web) and provides grounded answers with precise, verifiable citations.

## 🏗️ Architecture & Model Strategy

The system is built on a **Cloud-Hybrid Stack** to maximize portability and stability:

- **LLM Engine**: [Groq](https://groq.com/) (Llama 3.3 70B) for sub-second reasoning and high-fidelity generation.
- **Embedding Pipeline**: [BGE-Small-EN-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) via Hugging Face Inference API.
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) optimized for fast semantic retrieval.
- **Backend**: FastAPI (Python) with background task processing for multi-file ingestion.
- **Frontend**: Vanilla JS + TailwindCSS with a premium "Situation Room" glassmorphism aesthetic.

## 🚀 Key Features

- **Precise Citations**: Automatically maps answers to exact locations (Page numbers for PDFs, Tab/Row for Excel).
- **Two-Layered Defense**: Combines metadata-level retrieval filtering with strict system prompting to eliminate hallucinations.
- **Multi-Format Support**: 
  - **PDF**: Deterministic text stream extraction.
  - **Excel**: Row-by-row grid context mapping.
  - **Word**: Section-based heading extraction.
  - **Web**: Clean BS4 scraping with script/style decomposition.
- **Token Optimization**: Forced JSON-only messaging protocol to minimize API costs and latency.

## 🛠️ Setup Instructions

### Local Development
1. Clone the repository.
2. Create a `.env` file based on the template:
   ```env
   GROQ_API_KEY=your_key
   HUGGINGFACE_API_KEY=your_key
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the backend:
   ```bash
   python backend/main.py
   ```
5. Open `frontend/index.html` in your browser.

### Docker Deployment
Run the following command to launch the entire stack:
```bash
docker-compose up --build
```

## 📜 License
Internal Technical Assessment - ANSR
