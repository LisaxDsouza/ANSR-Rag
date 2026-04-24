# Architectural Diagram

```mermaid
graph TD
    subgraph "1. INGESTION LAYER"
        U[User Upload / URL] --> P{Format Detector}
        P -->|PDF| P1[PyPDF2 Stream Parser]
        P -->|Excel| P2[Pandas Grid Context Mapper]
        P -->|Docx/Web| P3[BS4 Section Parser]
        P1 & P2 & P3 --> C[Recursive Chunking]
    end

    subgraph "2. STORAGE & INDEXING"
        C --> E[BGE-Small Embedding Model]
        E --> V[FAISS Vector Index]
        C --> B[BM25 Keyword Index]
        V & B --> M[(Metadata Registry JSON)]
    end

    subgraph "3. RETRIEVAL & CITATION ENGINE"
        Q[User Query] --> Filter{Resource Filter}
        Filter -->|Selected IDs| H[Hybrid Search RRF]
        H -->|Semantic| V
        H -->|Keyword| B
        H --> Chunks[Top K Context Chunks]
    end

    subgraph "4. ORCHESTRATION & GENERATION"
        Chunks --> L[Llama 3.3 70B Engine]
        L --> J{JSON Schema Validator}
        J -->|Valid| A[Answer + Citation Badge]
        J -->|Invalid| R[Retry/Fallback]
    end

    U -.->|Select/Unselect| Filter
```

## Data Flow Narrative:
1.  **Ingestion**: Files are decomposed into chunks while preserving **lineage metadata** (page, row, tab).
2.  **Hybrid Search**: The system runs dual-retrieval (Semantic + Keyword) and merges results via **Reciprocal Rank Fusion (RRF)**.
3.  **Grounded QA**: The LLM (Llama 3.3) is constrained via **System Instructions** to only use the provided chunks.
4.  **Citation Mapping**: The "Citation Engine" pulls the specific metadata from the retrieved chunk and formats it into the UI badge.
