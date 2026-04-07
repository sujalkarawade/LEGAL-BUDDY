from fastapi import APIRouter, HTTPException
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from pydantic import BaseModel

import backend.state as state
from app.analysis import build_summary_docs
from app.llm import invoke_with_groq_fallback
from app.prompts import QA_PROMPT, SUMMARY_PROMPT

router = APIRouter()


class QuestionRequest(BaseModel):
    question: str


@router.post("/summarize")
def summarize():
    if not state.final_docs:
        raise HTTPException(status_code=400, detail="Embed a document first.")
    try:
        pdf_docs = [d for d in state.final_docs if d.metadata.get("source") == "PDF"] or state.final_docs
        summary_docs = build_summary_docs(pdf_docs)
        response = invoke_with_groq_fallback(
            lambda llm: create_stuff_documents_chain(llm, SUMMARY_PROMPT).invoke({"context": summary_docs})
        )
        answer = response.get("answer", response) if isinstance(response, dict) else response
        return {"summary": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ask")
def ask(body: QuestionRequest):
    if state.vectors is None:
        raise HTTPException(status_code=400, detail="Embed a document first.")
    try:
        retriever = state.vectors.as_retriever()
        response = invoke_with_groq_fallback(
            lambda llm: create_retrieval_chain(
                retriever,
                create_stuff_documents_chain(llm, QA_PROMPT),
            ).invoke({"input": body.question})
        )
        state.groq_model = response.get("groq_model", state.groq_model)
        chunks = [
            {"source": d.metadata.get("source", "Unknown"), "content": d.page_content}
            for d in response.get("context", [])
        ]
        return {"answer": response["answer"], "chunks": chunks}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
