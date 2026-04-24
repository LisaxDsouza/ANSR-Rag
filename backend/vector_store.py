import os
import json
import numpy as np
import requests
import time
import faiss
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class VectorStoreManager:
    def __init__(self):
        self.storage_dir = os.getenv("CHROMA_DB_PATH", "./backend/vector_db")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.index_path = os.path.join(self.storage_dir, "faiss.index")
        self.meta_path = os.path.join(self.storage_dir, "metadata.json")
        
        # FAISS Index Configuration (Flat L2 for exact search at this scale)
        self.dimension = 384 # Dimension for BGE-Small
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'r') as f:
                self.metadata_store = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dimension) # Inner Product for Cosine Similarity
            self.metadata_store = []
        
        # Hugging Face Inference API Configuration
        self.hf_token = os.getenv("HUGGINGFACE_API_KEY", "")
        self.model_id = "BAAI/bge-small-en-v1.5"
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
        # Initialize text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def _get_embeddings(self, texts):
        """Calls Hugging Face Inference API for embeddings."""
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=20)
            if response.status_code == 200:
                # Normalize for Cosine Similarity (Inner Product)
                embeddings = np.array(response.json()).astype('float32')
                faiss.normalize_L2(embeddings)
                return embeddings
            else:
                raise Exception(f"HF API Error: {response.status_code}")
        except Exception as e:
            print(f"Embedding Error: {e}")
            return np.zeros((len(texts), self.dimension)).astype('float32')

    def add_documents(self, parsed_content, doc_id, filename):
        """Chunks parsed content and adds it to the FAISS index."""
        all_chunks = []
        all_metadatas = []
        
        for item in parsed_content:
            text = item["text"]
            base_metadata = item["metadata"]
            chunks = self.splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                meta = base_metadata.copy()
                meta.update({"doc_id": doc_id, "filename": filename, "content": chunk})
                all_metadatas.append(meta)
        
        if not all_chunks:
            return 0

        # Batch processing for embeddings
        batch_size = 32
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings = self._get_embeddings(batch)
            self.index.add(embeddings)
            self.metadata_store.extend(all_metadatas[i:i + batch_size])
        
        # Persist
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'w') as f:
            json.dump(self.metadata_store, f)
            
        return len(all_chunks)

    def query(self, query_text, selected_doc_ids, n_results=5):
        """Queries the FAISS index with metadata filtering."""
        if self.index.ntotal == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}

        # 1. Get query embedding
        query_embedding = self._get_embeddings([query_text])
        
        # 2. Search (Top-K) - We search a bit more to handle filtering
        D, I = self.index.search(query_embedding, min(self.index.ntotal, 100))
        
        final_docs = []
        final_metadatas = []
        
        # 3. Filter by selected_doc_ids
        for idx in I[0]:
            if idx == -1: continue
            meta = self.metadata_store[idx]
            if meta["doc_id"] in selected_doc_ids:
                final_docs.append(meta["content"])
                final_metadatas.append(meta)
                if len(final_docs) >= n_results:
                    break
            
        return {
            "ids": [[""] * len(final_docs)],
            "documents": [final_docs],
            "metadatas": [final_metadatas]
        }
