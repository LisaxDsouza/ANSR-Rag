# ANSR Knowledge Assistant - Design Document

## 1. Data Processing & Retrieval (RAG)

### Structured Data Handling (Excel & Web)
- **Excel Strategy**: Instead of flat text extraction, we use a **Grid-to-Context Mapping**. Each row is transformed into a semantically rich sentence: *"In Sheet [Name], Row [Number]: [Header] is [Value]."*. This preserves the structural relationships that are lost in standard RAG.
- **Web Strategy**: We use **Selective Decomposition**. We strip all non-semantic tags (scripts, styles, navs) using BeautifulSoup4, then segment the body text by headers to maintain the "Section" metadata required for citations.

### Chunking & Citation Mapping
- **Strategy**: We use **Recursive Character Text Splitting** with a 1000-character window and 20% overlap.
- **Lineage Tracking**: Each chunk is an object containing `text` + `metadata`. The metadata stores the "Physical Location" (Page for PDF, Row for Excel). During retrieval, this metadata travels with the chunk to the LLM, enabling zero-guess citations.

### State Management & Latency
- **UI Level**: The frontend maintains a `selected_doc_ids` array.
- **Database Level**: We use **Metadata Filtering** in FAISS. When a query is made, FAISS performs a pre-filter or a post-filter to only consider vectors belonging to the selected IDs. This ensures sub-second latency regardless of the total Hub size (40 or 4,000 docs).

## 2. System Components

### High-Speed Retrieval
- **Vector Database**: We use **FAISS (Facebook AI Similarity Search)**. It is an industry-standard library for dense vector clustering and search. 
- **Hybrid Engine**: We combine FAISS with **BM25 (Keyword Search)** using **Reciprocal Rank Fusion (RRF)**. This ensures that exact acronyms or project names (like "ANSR") are always found, even if the semantic embedding is slightly off.

### Security & Sessions
- **Authentication**: For an internal tool, we recommend integration with **Azure AD or Okta (OIDC)**.
- **Session Management**: We suggest using **Redis-backed session stores** to maintain chat history and selected resource state across browser refreshes.

## 3. The AI Strategy

### Model Recommendation
- **Primary LLM**: **Llama 3.3 70B** (via Groq).
- **Reasoning**: It provides GPT-4 level reasoning for complex document analysis but with much lower latency (sub-second generation) and lower operational cost.

### Prompt Engineering & Citations
- **Strategy**: We use a **Strict JSON Messaging Protocol**. The system instructions force the LLM to output a JSON schema. If the LLM cannot find a direct quote, it is instructed to return a specific "not_found" flag.
- **System Instruction**: *"You are a grounded assistant. Use ONLY the provided context. If the answer is missing, state 'Information not found.' Always provide 'quote', 'source', and 'location' keys."*

## 4. Cost & Scaling

### Minimizing "Token Waste"
- **Context Pruning**: We only send the top 5 most relevant chunks to the LLM. 
- **Token Budgeting**: We use dynamic truncation to ensure we never exceed the LLM's efficient context window, avoiding expensive "long-context" pricing.

### Scaling to 4,000+ Documents
- **Sharding**: FAISS indices can be sharded by department or document category.
- **Distributed Indexing**: The ingestion pipeline would move to a message queue (Celery/RabbitMQ) to handle bulk uploads in parallel across multiple worker nodes.
