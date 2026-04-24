# Knowledge Assistant Architecture & Design

This document serves as the "Source of Truth" for the architecture, data structures, and implementation details of the Knowledge Assistant RAG tool.

## 1. Technical Stack (The "4-Hour" OSS Stack)
- **Backend**: FastAPI (Python)
- **Vector Database**: ChromaDB (Local, Persistent)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (Local via Sentence-Transformers)
- **LLM Engine**: Groq (Llama 3 70B/8B) or Ollama (Local)
- **Frontend**: HTML5 + TailwindCSS + JavaScript (Fetch API)

## 2. Data Ingestion Pipeline
Each file type is parsed to extract both text and **Location Metadata**.

| Source | Parser | Metadata Key | Location Format |
| :--- | :--- | :--- | :--- |
| **PDF** | `PyPDF2` | `page` | "Page X" |
| **Excel** | `pandas` | `tab_row` | "Tab: [Sheet], Row: [Index]" |
| **Word** | `python-docx` | `section` | "Section: [Heading Text]" |
| **Web** | `BeautifulSoup` | `section` | "Section: [Nearest Header]" |

### Chunking Strategy (Efficiency & Context)
To maintain contextual integrity while ensuring sub-second retrieval:
- **Algorithm**: `RecursiveCharacterTextSplitter` (via LangChain).
- **Chunk Size**: 1,000 characters.
- **Chunk Overlap**: 200 characters (20%).
- **Rationale**: Recursive splitting preserves paragraph and sentence boundaries, which is critical for providing clean "Direct Quotes" in citations. The 20% overlap ensures that facts split across boundaries are still retrievable.

## 3. Knowledge Hub & Metadata Strategy (40 Documents)
To manage up to 40 documents with high-speed filtering:

### Document Registry
The system maintains a registry (in-memory or `docs.json`) mapping IDs to metadata:
```json
{
  "doc_id": "uuid-1234",
  "filename": "Product_Specs.pdf",
  "type": "pdf",
  "uploaded_at": "2024-04-24T12:00:00Z"
}
```

### Filtering Logic
ChromaDB chunks are stored with the `doc_id`. Queries are restricted using the `where` clause:
```python
results = collection.query(
    query_texts=[user_query],
    n_results=5,
    where={"doc_id": {"$in": selected_doc_ids}}
)
```

## 4. Grounded QA & Citation Engine

### Grounded QA Strategy (The "Two-Layered Defense")
To ensure the AI strictly adheres to the selected resources:
1. **Layer 1: Hard Metadata Filter**: All vector searches are restricted using the `where={"doc_id": {"$in": selected_ids}}` clause. The LLM context never contains "unselected" data.
2. **Layer 2: Strict System Instruction**: The LLM is prompted to only use the provided context. If the answer is missing, it must explicitly state so rather than using internal knowledge.
3. **Layer 3: Verification Loop**: Every answer must be mapped to a `quote` and `location`. If the LLM cannot provide these, the answer is discarded.

### Prompt Strategy
The LLM is instructed to return a structured response (preferably JSON) to ensure citation accuracy:
```json
{
  "answer": "The product supports 5G connectivity.",
  "citation": {
    "quote": "Connectivity: Integrated 5G modem...",
    "source": "Product_Specs.pdf",
    "location": "Page 4"
  }
}
```

## 5. UI/UX Specifications
- **Sidebar (Resource Manager)**:
  - Search bar to filter the 40 documents.
  - "Select All" / "Clear All" toggles.
  - Visual icons for file types.
  - Checkboxes to control the query scope.
- **Chat Interface**:
  - Glassmorphism design (blur backgrounds, subtle borders).
  - Streamed responses (simulated or real).
  - Clickable citations that highlight or point to the source metadata.

## 6. Scaling & Performance
- **Local Embeddings**: `bge-small` ensures sub-second vectorization.
- **RAG Pattern**: Only relevant chunks are passed to the LLM, keeping token usage low and preventing context window overflow.
- **Future Growth**: The `doc_id` filtering strategy scales linearly to 4,000+ documents.

## 7. Infrastructure & Hosting (Production-Ready)
The application is designed to be fully containerized using **Docker Compose** for maximum portability between on-premise and cloud environments.

### Service Architecture
| Service | Technology | Role |
| :--- | :--- | :--- |
| **`web`** | FastAPI (Python) | Ingestion logic, RAG orchestration, Frontend delivery. |
| **`chroma`** | ChromaDB | Standalone vector database server for semantic search. |
| **`db`** | PostgreSQL | Relational storage for Document Registry, sessions, and logs. |

### Persistence & Portability
- **Docker Volumes**: Used for `/data` (Chroma vectors) and `/pgdata` (Postgres records) to ensure data persistence across restarts.
- **Environment Config**: A `.env` file manages provider-specific settings (e.g., Groq API keys, DB credentials).
- **Scalability**: By separating the vector store from the web server, the system can scale to 4,000+ documents by simply increasing the resources allocated to the `chroma` service.

## 8. Cost/Quality Balance (Tiered Strategy)
To satisfy the requirement of high reasoning quality at low operational costs:
- **Phase 1: Deterministic Parsing**: Use `PyPDF2`, `pandas`, and `python-docx` for 0-cost, high-speed ingestion.
- **Phase 2: Light Embeddings**: Use `bge-small` (local) for free, sub-second vectorization.
- **Phase 3: Tiered Generation**: 
  - Use **Llama 3 8B** for 90% of queries (fast, cheap).
  - Reserved **Llama 3 70B** or **GPT-4o-mini** only for complex cross-document synthesis.

## 9. Token Efficiency Protocol (Groq Optimization)
To minimize token utilization and ensure "on-point" responses:
- **Strict JSON Messaging**: LLM is instructed to skip greetings and conversational filler, returning ONLY structured JSON.
- **Hard Max-Token Cap**: Response length is hard-capped at 300 tokens in the API call.
- **Context Pruning**: Only the top-k relevant chunks (max 5) are sent to the LLM, with a total input cap of 2,000 tokens.
- **Frontend Rendering**: Raw JSON is parsed by the JavaScript engine and rendered into a premium, styled chat interface (Markdown + Citation Badges), ensuring the "WOW" factor remains high while keeping the "under-the-hood" communication lean.
