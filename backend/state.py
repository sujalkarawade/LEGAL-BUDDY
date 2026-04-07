"""In-memory session state shared across routers (single-user dev server)."""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

vectors: FAISS | None = None
final_docs: list[Document] = []
embedding_backend: str = ""
groq_model: str = ""
