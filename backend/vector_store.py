import os
import json
import numpy as np
import requests
import time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class VectorStoreManager:
    def __init__(self):
        self.storage_path = os.getenv("CHROMA_DB_PATH", "./backend/vector_storage.json")
        # Ensure path is a JSON file for our lite version
        if os.path.isdir(self.storage_path):
            self.storage_path = os.path.join(self.storage_path, "vectors.json")
            
        self.data = self._load_storage()
        
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

    def _load_storage(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

    def _save_storage(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self.data, f)

    def _get_embeddings(self, texts):
        """Calls Hugging Face Inference API for embeddings."""
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            # Fallback for API errors (returns dummy vectors for testing if API fails)
            print(f"HF API Error: {response.status_code}. Using dummy vectors.")
            return [[0.0] * 384 for _ in texts]

    def add_documents(self, parsed_content, doc_id, filename):
        """Chunks parsed content and adds it to our Lite Vector Store."""
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for item in parsed_content:
            text = item["text"]
            base_metadata = item["metadata"]
            chunks = self.splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                all_chunks.append(chunk)
                meta = base_metadata.copy()
                meta.update({"doc_id": doc_id, "filename": filename})
                all_metadatas.append(meta)
                all_ids.append(chunk_id)
        
        # Batch processing for embeddings
        batch_size = 32
        embeddings = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings.extend(self._get_embeddings(batch))
        
        # Update local storage
        self.data["ids"].extend(all_ids)
        self.data["embeddings"].extend(embeddings)
        self.data["documents"].extend(all_chunks)
        self.data["metadatas"].extend(all_metadatas)
        
        self._save_storage()
        return len(all_chunks)

    def query(self, query_text, selected_doc_ids, n_results=5):
        """Performs Cosine Similarity search with metadata filtering using Numpy."""
        if not self.data["embeddings"]:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}

        # 1. Get query embedding
        query_embedding = np.array(self._get_embeddings([query_text])[0])
        
        # 2. Filter indices by selected_doc_ids
        indices = [i for i, meta in enumerate(self.data["metadatas"]) if meta["doc_id"] in selected_doc_ids]
        
        if not indices:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}

        # 3. Calculate similarities for filtered items
        filtered_embeddings = np.array([self.data["embeddings"][i] for i in indices])
        
        # Cosine Similarity = (A . B) / (||A|| * ||B||)
        # Normalize vectors for easier calculation
        norm_filtered = filtered_embeddings / np.linalg.norm(filtered_embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        
        similarities = np.dot(norm_filtered, norm_query)
        
        # 4. Get Top-K
        top_indices_local = np.argsort(similarities)[::-1][:n_results]
        
        final_ids = []
        final_docs = []
        final_metadatas = []
        
        for idx in top_indices_local:
            global_idx = indices[idx]
            final_ids.append(self.data["ids"][global_idx])
            final_docs.append(self.data["documents"][global_idx])
            final_metadatas.append(self.data["metadatas"][global_idx])
            
        return {
            "ids": [final_ids],
            "documents": [final_docs],
            "metadatas": [final_metadatas]
        }
