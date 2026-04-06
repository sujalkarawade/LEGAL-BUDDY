import time

import streamlit as st
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

from app.analysis import (
    advice_on_missing_clauses,
    advice_on_unusual_combinations,
    build_summary_docs,
    get_clause_risk,
    get_top_lawyers,
    show_cooccurrence_matrix,
)
from app.config import LAWYER_PATH, RAG_PATH, UPLOAD_DIR, initialize_runtime
from app.data import load_json_list, load_risk_data
from app.document_processing import vector_embedding
from app.llm import invoke_with_groq_fallback
from app.prompts import QA_PROMPT, SUMMARY_PROMPT
from app.ui import inject_global_styles, render_footer, render_page_intro, render_sidebar_status

initialize_runtime()

st.set_page_config(page_title="Legal Document Assistant", layout="wide")
inject_global_styles()
render_page_intro()

civil_rag = load_json_list(RAG_PATH, key="civil_law_data")
lawyer_rag = load_json_list(LAWYER_PATH)
risk_df = load_risk_data()

render_sidebar_status()

uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])
file_path = None

if uploaded_file:
    UPLOAD_DIR.mkdir(exist_ok=True)
    file_path = UPLOAD_DIR / uploaded_file.name
    with file_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())
    st.success("Document uploaded successfully.")
    st.caption(
        "Embeddings use larger chunks now to reduce OpenRouter cost and rate-limit issues whenever possible."
    )

if uploaded_file and file_path and st.button("Embed Document"):
    try:
        vector_embedding(file_path, civil_rag)
        backend = st.session_state.get("embedding_backend", "Unknown")
        st.success(f"Vector store is ready using {backend} embeddings.")
    except Exception as exc:
        st.error(f"Could not build embeddings: {exc}")

if uploaded_file and st.button("Summarize Document"):
    if "final_docs" not in st.session_state:
        st.warning("Embed the document first.")
    else:
        try:
            all_docs: list[Document] = st.session_state.final_docs
            pdf_docs = [doc for doc in all_docs if doc.metadata.get("source") == "PDF"] or all_docs
            summary_docs = build_summary_docs(pdf_docs)

            response = invoke_with_groq_fallback(
                lambda llm: create_stuff_documents_chain(llm, SUMMARY_PROMPT).invoke(
                    {"context": summary_docs}
                )
            )
            st.subheader("Document Summary")
            if isinstance(response, dict):
                st.write(response.get("answer", response))
            else:
                st.write(response)
        except Exception as exc:
            st.error(f"Could not summarize the document: {exc}")

prompt1 = st.text_input("Ask a question about the document")
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Embed the document first.")
    else:
        try:
            retriever = st.session_state.vectors.as_retriever()

            start = time.process_time()
            response = invoke_with_groq_fallback(
                lambda llm: create_retrieval_chain(
                    retriever,
                    create_stuff_documents_chain(llm, QA_PROMPT),
                ).invoke({"input": prompt1})
            )
            st.write("Response time:", round(time.process_time() - start, 2), "seconds")

            st.subheader("Answer")
            st.write(response["answer"])

            with st.expander("Relevant Document Chunks"):
                for doc in response["context"]:
                    source = doc.metadata.get("source", "Unknown")
                    st.write(f"Source: {source}")
                    st.write(doc.page_content)
                    st.write("---")
        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")

if "final_docs" in st.session_state:
    show_cooccurrence_matrix(st.session_state.final_docs)

    all_detected = set()
    for doc in st.session_state.final_docs:
        all_detected.update(doc.metadata.get("clauses", []))

    advice1 = advice_on_missing_clauses(all_detected)
    advice2 = advice_on_unusual_combinations(all_detected)
    st.subheader("Contract Advice")
    st.write(advice1)
    if advice2:
        st.write(advice2)

    risky = [clause for clause in all_detected if get_clause_risk(clause, risk_df) == "high"]
    if risky:
        st.subheader("High Risk Clauses Detected")
        st.write("Clauses: " + ", ".join(risky))
        top_lawyers = get_top_lawyers(lawyer_rag, area="Civil", top_n=3)
        st.subheader("Top Civil Lawyers Recommended")
        for lawyer in top_lawyers:
            st.write(f"{lawyer['name']} ({lawyer['location']}, {lawyer['experience']} yrs)")
        st.stop()

render_footer()
