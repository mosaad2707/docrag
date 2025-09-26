import os
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutil
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from rag_core.rag_service import RAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger("RAGService")

app = FastAPI(
    title="Production RAG Service API",
    description="API for document upload, indexing, and querying.",
    version="1.0.0"
)

# Singleton RAGService
rag_service = RAGService()

# --- Temporary File Storage ---
TEMP_DIR = "temp_uploads"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    context: List[Dict[str, Any]]
    message: str

class UploadResponse(BaseModel):
    filename: str
    session_id: str
    message: str

@app.post("/upload/", response_model=UploadResponse)
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    # log.info(f"Received upload request | filename={file.filename} session_id={session_id}")
    
    temp_file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # log.info(f"File saved temporarily | path={temp_file_path}")

        await rag_service.upload_and_index_document(temp_file_path, session_id)
        
        return UploadResponse(
            filename=file.filename,
            session_id=session_id,
            message="Document processed and indexed successfully."
        )
    except FileNotFoundError as e:
        log.error(f"File not found during processing | error={e}")
        raise HTTPException(status_code=404, detail=f"File error: {e}")
    except Exception as e:
        log.exception("Unexpected error during file upload")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            # log.info(f"Temporary file cleaned up | path={temp_file_path}")

@app.post("/query/", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    # log.info(f"Received query request | query={request.query} session_id={request.session_id}")
    
    try:
        context = await rag_service.query(request.query, request.session_id)
        if not context:
            return QueryResponse(
                context=[],
                message="No relevant context found for your query. Try uploading a relevant document first."
            )
            
        return QueryResponse(
            context=context,
            message="Context retrieved successfully."
        )
    except Exception as e:
        log.exception("Unexpected error during query")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.get("/")
def read_root():
    return {"status": "RAG Service is running"}