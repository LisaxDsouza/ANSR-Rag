import os
import json
import numpy as np
import requests
import time
import faiss
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class VectorStoreManager:
    def __init__(self):
        self.storage_dir = os.getenv("CHROMA_DB_PATH", "./backend/vector_db")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.index_path = os.path.join(self.storage_dir, "faiss.index")
        self.meta_path = os.path.join(self.storage_dir, "metadata.json")
        
        self.dimension = 384
        self.metadata_store = []
        
        # Initialize or Load FAISS
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'r') as f:
                self.metadata_store = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata_store = []
            
        # Initialize BM25 if we have data
        self.bm25 = None
        if self.metadata_store:
            self._update_bm25()
        
        # HF API Config
        self.hf_token = os.getenv("HUGGINGFACE_API_KEY", "")
        self.model_id = "BAAI/bge-small-en-v1.5"
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def _update_bm25(self):
        """Prepares the BM25 index for keyword search."""
        corpus = [m["content"].lower().split() for m in self.metadata_store]
        self.bm25 = BM25Okapi(corpus)

    def _get_embeddings(self, texts):
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=20)
            if response.status_code == 200:
                embeddings = np.array(response.json()).astype('float32')
                faiss.normalize_L2(embeddings)
                return embeddings
            return np.zeros((len(texts), self.dimension)).astype('float32')
        except:
            return np.zeros((len(texts), self.dimension)).astype('float32')

    def add_documents(self, parsed_content, doc_id, filename):
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
        
        if not all_chunks: return 0

        batch_size = 32
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings = self._get_embeddings(batch)
            self.index.add(embeddings)
            self.metadata_store.extend(all_metadatas[i:i + batch_size])
        
        self._update_bm25()
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'w') as f:
            json.dump(self.metadata_store, f)
            
        return len(all_chunks)

    def query(self, query_text, selected_doc_ids, n_results=5):
        """Hybrid Search: Merges FAISS (Semantic) and BM25 (Keyword) using RRF."""
        if self.index.ntotal == 0 or not selected_doc_ids:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}

        # 1. Filter indices by selected documents first
        valid_indices = [i for i, m in enumerate(self.metadata_store) if m["doc_id"] in selected_doc_ids]
        if not valid_indices:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}

        # 2. Semantic Search (FAISS)
        query_embedding = self._get_embeddings([query_text])
        D, I = self.index.search(query_embedding, min(self.index.ntotal, 50))
        
        semantic_ranks = {idx: rank for rank, idx in enumerate(I[0]) if idx in valid_indices}

        # 3. Keyword Search (BM25)
        tokenized_query = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Sort BM25 results only for valid documents
        keyword_indices = sorted(valid_indices, key=lambda i: bm25_scores[i], reverse=True)[:50]
        keyword_ranks = {idx: rank for rank, idx in enumerate(keyword_indices)}

        # 4. Reciprocal Rank Fusion (RRF)
        # Score = sum( 1 / (k + rank) )
        k = 60
        combined_scores = {}
        all_candidate_indices = set(semantic_ranks.keys()) | set(keyword_ranks.keys())
        
        for idx in all_candidate_indices:
            score = 0
            if idx in semantic_ranks:
                score += 1.0 / (k + semantic_ranks[idx])
            if idx in keyword_ranks:
                score += 1.0 / (k + keyword_ranks[idx])
            combined_scores[idx] = score

        # 5. Final Top-K
        top_indices = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)[:n_results]
        
        final_docs = [self.metadata_store[i]["content"] for i in top_indices]
        final_metadatas = [self.metadata_store[i] for i in top_indices]
            
        return {
            "ids": [[""] * len(final_docs)],
            "documents": [final_docs],
            "metadatas": [final_metadatas]
        }
