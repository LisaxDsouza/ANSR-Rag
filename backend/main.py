import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from parsers import get_parser, parse_web
from vector_store import VectorStoreManager
from engine import RAGEngine

load_dotenv()

app = FastAPI(title="Knowledge Assistant API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./backend/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit

# Initialize Managers
vector_store = VectorStoreManager()
rag_engine = RAGEngine()

# Persistent Registry
REGISTRY_PATH = os.path.join(UPLOAD_DIR, "registry.json")

def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return []

def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f)

document_registry = load_registry()

class DocumentInfo(BaseModel):
    id: str
    filename: str
    status: str
    type: str

class QueryRequest(BaseModel):
    query: str
    selected_doc_ids: List[str]

@app.post("/upload", response_model=List[DocumentInfo])
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    uploaded_docs = []
    for file in files:
        # Check file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File {file.filename} exceeds the 10MB limit.")

        # Check for duplicates (basic filename check for now)
        is_duplicate = any(doc["filename"] == file.filename for doc in document_registry)
        if is_duplicate:
            # We allow it but maybe suffix it or warn
            logger.info(f"Duplicate filename uploaded: {file.filename}")

        doc_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        doc_info = {
            "id": doc_id,
            "filename": file.filename,
            "status": "processing",
            "path": file_path,
            "type": "file"
        }
        document_registry.append(doc_info)
        save_registry(document_registry)
        uploaded_docs.append(DocumentInfo(id=doc_id, filename=file.filename, status="processing", type="file"))
        
        # Trigger background ingestion
        background_tasks.add_task(process_document, doc_id, file_path, file.filename)
        
    return uploaded_docs

@app.post("/add-url", response_model=DocumentInfo)
async def add_url(background_tasks: BackgroundTasks, url: str):
    doc_id = str(uuid.uuid4())
    doc_info = {
        "id": doc_id,
        "filename": url,
        "status": "processing",
        "path": url,
        "type": "url"
    }
    document_registry.append(doc_info)
    save_registry(document_registry)
    
    # Trigger background ingestion
    background_tasks.add_task(process_url, doc_id, url)
    
    return DocumentInfo(id=doc_id, filename=url, status="processing", type="url")

async def process_document(doc_id, file_path, filename):
    try:
        parser = get_parser(filename)
        if parser:
            content = parser(file_path)
            if content:
                vector_store.add_documents(content, doc_id, filename)
                # Update registry status
                for doc in document_registry:
                    if doc["id"] == doc_id:
                        doc["status"] = "ready"
                        save_registry(document_registry)
                        return
            else:
                raise Exception("No text content could be extracted")
        else:
            raise Exception("Unsupported file format")
    except Exception as e:
        logger.error(f"Ingestion Error: {e}")
        for doc in document_registry:
            if doc["id"] == doc_id:
                doc["status"] = "error"
                save_registry(document_registry)
                break

async def process_url(doc_id, url):
    try:
        content = parse_web(url)
        if content:
            vector_store.add_documents(content, doc_id, url)
            # Update registry status
            for doc in document_registry:
                if doc["id"] == doc_id:
                    doc["status"] = "ready"
                    save_registry(document_registry)
                    return
        else:
            raise Exception("Could not extract content from URL")
    except Exception as e:
        logger.error(f"URL Scrape Error: {e}")
        for doc in document_registry:
            if doc["id"] == doc_id:
                doc["status"] = "error"
                save_registry(document_registry)
                break

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    return [DocumentInfo(id=doc["id"], filename=doc["filename"], status=doc["status"], type=doc["type"]) for doc in document_registry]

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    global document_registry
    doc_to_delete = None
    for doc in document_registry:
        if doc["id"] == doc_id:
            doc_to_delete = doc
            break
            
    if not doc_to_delete:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # 1. Remove from registry
    document_registry = [doc for doc in document_registry if doc["id"] != doc_id]
    save_registry(document_registry)
    
    # 2. Remove file from disk (if it's a file)
    if doc_to_delete["type"] == "file" and os.path.exists(doc_to_delete["path"]):
        try:
            os.remove(doc_to_delete["path"])
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            
    return {"message": "Document deleted successfully"}

@app.post("/query")
async def handle_query(request: QueryRequest):
    if not request.selected_doc_ids:
        raise HTTPException(status_code=400, detail="No documents selected")
        
    # 1. Retrieval
    results = vector_store.query(request.query, request.selected_doc_ids)
    
    # Check if results exist
    if not results['documents'] or not results['documents'][0]:
        return {"answer": "No relevant information found in the selected documents.", "citation": None}
    
    # 2. Format chunks for the engine
    context_chunks = []
    for i in range(len(results['documents'][0])):
        context_chunks.append({
            "content": results['documents'][0][i],
            "metadata": results['metadatas'][0][i]
        })
        
    # 3. Generation
    answer = rag_engine.generate_answer(request.query, context_chunks)
    return answer

@app.get("/")
async def root():
    return {"message": "Knowledge Assistant API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
