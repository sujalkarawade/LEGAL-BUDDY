from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

import backend.state as state
from app.config import RAG_PATH, UPLOAD_DIR
from app.data import load_json_list
from app.document_processing import vector_embedding

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    UPLOAD_DIR.mkdir(exist_ok=True)
    file_path = UPLOAD_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)
    return {"filename": file.filename, "path": str(file_path)}


@router.post("/embed")
async def embed_document(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found. Upload it first.")

    civil_rag = load_json_list(RAG_PATH, key="civil_law_data")
    try:
        vectors, backend_name, final_docs = vector_embedding(file_path, civil_rag)
        state.vectors = vectors
        state.final_docs = final_docs
        state.embedding_backend = backend_name
        return {"backend": backend_name, "doc_count": len(final_docs)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
