from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.analysis import identify_clauses
from app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_EMBEDDING_MODEL,
    PDF_CHUNK_OVERLAP,
    PDF_CHUNK_SIZE,
)
from app.embeddings import (
    build_openrouter_embeddings,
)


def build_vector_store(all_docs: list[Document]) -> tuple[FAISS, str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured on the server.")
    vector_store = FAISS.from_documents(all_docs, build_openrouter_embeddings())
    return vector_store, f"OpenRouter ({OPENROUTER_EMBEDDING_MODEL})"


def vector_embedding(path: Path, civil_rag: list[dict]) -> tuple[FAISS, str, list[Document], int, int]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    page_count = len(docs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHUNK_SIZE,
        chunk_overlap=PDF_CHUNK_OVERLAP,
    )
    final_docs = splitter.split_documents(docs)

    def process_pdf_chunk(doc: Document) -> Document:
        metadata = dict(doc.metadata)
        metadata.update({"source": "PDF", "clauses": identify_clauses(doc.page_content)})
        return Document(page_content=doc.page_content, metadata=metadata)

    def process_rag_entry(entry: dict) -> Document | None:
        if entry.get("type") == "clause":
            text = entry["clause"] + "\n" + entry["layman_terms"]
        elif entry.get("type") == "term":
            text = entry["term"] + "\n" + entry["layman_example"]
        else:
            return None
        return Document(
            page_content=text,
            metadata={"source": "RAG", "clauses": identify_clauses(text)},
        )

    with ThreadPoolExecutor() as executor:
        pdf_docs = list(executor.map(process_pdf_chunk, final_docs))
        rag_docs = list(executor.map(process_rag_entry, civil_rag))
        rag_docs = [doc for doc in rag_docs if doc is not None]

    all_docs = pdf_docs + rag_docs
    
    unique_clauses = set()
    for d in pdf_docs:
        unique_clauses.update(d.metadata.get("clauses", []))
    clause_count = len(unique_clauses)
    
    vectors, backend_name = build_vector_store(all_docs)
    return vectors, backend_name, all_docs, page_count, clause_count
