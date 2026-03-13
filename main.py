import asyncio
import hashlib
import itertools
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nest_asyncio
import numpy as np
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploaded_docs"
RAG_PATH = BASE_DIR / "civil_law(RAG).json"
LAWYER_PATH = BASE_DIR / "lawyer(RAG).json"
RISK_PATH = BASE_DIR / "legal_contract_clauses.csv"

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODEL_CANDIDATES = [
    os.getenv("GROQ_MODEL", "").strip(),
    DEFAULT_GROQ_MODEL,
    "llama-3.3-70b-versatile",
]
EMBEDDING_MODEL = "gemini-embedding-001"
PDF_CHUNK_SIZE = 4000
PDF_CHUNK_OVERLAP = 400
FREE_TIER_EMBED_ITEM_BUDGET = 95
LOCAL_EMBEDDING_DIMENSION = 512


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
nest_asyncio.apply()


def get_config_value(name: str) -> str:
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        return os.getenv(name, "").strip()


GROQ_API_KEY = get_config_value("GROQ_API_KEY")
GOOGLE_API_KEY = get_config_value("GOOGLE_API_KEY")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


qa_prompt = ChatPromptTemplate.from_template(
    """
You are a legal document assistant.
- Summarize content in simple language.
- Identify potential risks.
- Explain complex clauses.
Answer only from the provided context.
<context>
{context}
</context>
Question: {input}
"""
)

summary_prompt = ChatPromptTemplate.from_template(
    """
Summarize this legal document in simple, everyday language.
Cover the important legal terms and clauses while summarizing.
Explain each legal clause in simple terms and highlight any risks mentioned in the document.
<context>
{context}
</context>
"""
)

clause_prompt = ChatPromptTemplate.from_template(
    """
Explain the following legal clause in simple terms and highlight any risks:
{input}
"""
)


class LocalHashedEmbeddings(Embeddings):
    """Offline fallback embeddings to avoid external quota limits."""

    def __init__(self, dimension: int = LOCAL_EMBEDDING_DIMENSION):
        self.dimension = dimension

    def _embed_text(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"\b\w+\b", text.lower())

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


CLAUSE_PATTERNS = [
    (r"\bparties\b|\bbetween\b", "PARTIES"),
    (r"\bobject\b|\bpurpose\b", "OBJECT_PURPOSE"),
    (r"\bconsideration\b|\bpayment\b|\bfees?\b|\brent\b", "CONSIDERATION_PAYMENT"),
    (r"\bterm\b|\bduration\b|\bvalidity\b", "TERM_DURATION"),
    (r"\bobligations?\b|\bduties\b|\bresponsibilit(y|ies)\b", "OBLIGATIONS"),
    (r"\bright(s)?\b|\bprivileges?\b", "RIGHTS"),
    (r"\btermination\b|\bcancellation\b|\brescind\b", "TERMINATION"),
    (r"\bliabilit(y|ies)\b|\bindemnif(y|ication)\b", "LIABILITY_INDEMNITY"),
    (r"\bconfidential(ity)?\b|\bsecrecy\b", "CONFIDENTIALITY"),
    (r"\bdispute\b|\barbitration\b|\bjurisdiction\b", "DISPUTE_RESOLUTION"),
    (r"\bgoverning\s+law\b|\bapplicable\s+law\b", "GOVERNING_LAW"),
    (r"\bforce\s+majeure\b|\bact\s+of\s+god\b", "FORCE_MAJEURE"),
    (r"\bnotice\b|\bcommunication\b", "NOTICE_COMMUNICATION"),
    (r"\bsign(ed|ature)?\b|\bexecution\b", "SIGNATURE_EXECUTION"),
]

EXPECTED_CLAUSES = ["TERM_DURATION", "RENT_PAYMENT", "SECURITY_DEPOSIT", "TERMINATION"]


def identify_clauses(text: str) -> list[str]:
    clauses = []
    for pattern, label in CLAUSE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            clauses.append(label)
    return clauses


@st.cache_data(show_spinner=False)
def load_json_list(path: Path, key: str | None = None):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if key:
        return data.get(key, [])
    return data


@st.cache_data(show_spinner=False)
def load_risk_data() -> pd.DataFrame:
    if not RISK_PATH.exists():
        return pd.DataFrame(columns=["clause_type", "risk_level"])
    return pd.read_csv(RISK_PATH)


@st.cache_resource(show_spinner=False)
def build_llm(model_name: str) -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY. Set it in the environment or Streamlit secrets.")
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)


@st.cache_resource(show_spinner=False)
def build_google_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not GOOGLE_API_KEY:
        raise RuntimeError("Missing GOOGLE_API_KEY. Set it in the environment or Streamlit secrets.")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)


@st.cache_resource(show_spinner=False)
def build_local_embeddings() -> LocalHashedEmbeddings:
    return LocalHashedEmbeddings()


def summarize_embedding_failure(exc: Exception) -> str:
    message = str(exc)
    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        return "Google embedding quota is exhausted for the current rate-limit window"
    if "PERMISSION_DENIED" in message or "403" in message:
        return "Google rejected the embedding request"
    if "NOT_FOUND" in message or "404" in message:
        return "Google embedding model is unavailable"
    return "Google embeddings are unavailable right now"


def groq_model_candidates() -> list[str]:
    seen = set()
    candidates = []
    for model_name in GROQ_MODEL_CANDIDATES:
        if model_name and model_name not in seen:
            seen.add(model_name)
            candidates.append(model_name)
    return candidates


def build_summary_docs(
    docs: list[Document],
    max_input_tokens: int = 3500,
    approx_chars_per_token: int = 4,
) -> list[Document]:
    """Trim document list so Groq token limits are not exceeded.

    Groq free tiers often allow around 12k input tokens; we stay under that
    by approximating \(1\) token \(\approx\) 4 characters and truncating.
    """
    char_limit = max_input_tokens * approx_chars_per_token
    selected: list[Document] = []
    total_chars = 0

    for doc in docs:
        if total_chars >= char_limit:
            break

        content = doc.page_content or ""
        remaining = char_limit - total_chars
        if remaining <= 0:
            break

        if len(content) <= remaining:
            selected.append(doc)
            total_chars += len(content)
        else:
            # Truncate the last document so we don't exceed the limit
            selected.append(
                Document(page_content=content[:remaining], metadata=doc.metadata)
            )
            total_chars += remaining
            break

    return selected


def invoke_with_groq_fallback(factory):
    last_error = None

    for model_name in groq_model_candidates():
        try:
            llm = build_llm(model_name)
            result = factory(llm)
            st.session_state.groq_model = model_name
            return result
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if "model_decommissioned" in message or "decommissioned" in message or "not supported" in message:
                continue
            if "404" in message and "model" in message:
                continue
            raise

    raise RuntimeError(
        "No working Groq model was available. Tried: "
        + ", ".join(groq_model_candidates())
        + f". Last error: {last_error}"
    )


def build_vector_store(all_docs: list[Document], prefer_local: bool = False) -> tuple[FAISS, str]:
    if GOOGLE_API_KEY and not prefer_local:
        try:
            vector_store = FAISS.from_documents(all_docs, build_google_embeddings())
            return vector_store, "Google Gemini"
        except Exception as exc:
            st.warning(f"{summarize_embedding_failure(exc)}. Falling back to local embeddings.")
    elif GOOGLE_API_KEY and prefer_local:
        st.info("Large upload detected. Using local embeddings to avoid Google free-tier quota issues.")
    else:
        st.info("GOOGLE_API_KEY not found. Using local embeddings.")

    vector_store = FAISS.from_documents(all_docs, build_local_embeddings())
    return vector_store, "Local fallback"


def vector_embedding(path: Path, civil_rag: list[dict]) -> None:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
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

    total_embed_items = len(pdf_docs) + len(rag_docs)
    all_docs = pdf_docs + rag_docs
    prefer_local = total_embed_items > FREE_TIER_EMBED_ITEM_BUDGET
    st.session_state.vectors, st.session_state.embedding_backend = build_vector_store(
        all_docs,
        prefer_local=prefer_local,
    )
    st.session_state.final_docs = all_docs


def get_all_detected_clauses(docs: list[Document]) -> list[list[str]]:
    all_clauses = []
    for doc in docs:
        clauses = doc.metadata.get("clauses", [])
        if clauses:
            all_clauses.append(clauses)
    return all_clauses


def show_cooccurrence_matrix(docs: list[Document]) -> None:
    all_clauses = get_all_detected_clauses(docs)
    pairs = []
    for clauses in all_clauses:
        pairs.extend(itertools.combinations(sorted(set(clauses)), 2))
    co_occurrence = Counter(pairs)
    if co_occurrence:
        df = (
            pd.DataFrame({(a, b): count for (a, b), count in co_occurrence.items()}, index=[0])
            .T.sort_values(0, ascending=False)
        )
        st.subheader("Clause Co-Occurrence Matrix")
        st.dataframe(df)
    else:
        st.info("No clause pairs detected for co-occurrence analysis.")


def advice_on_missing_clauses(
    detected_clauses: set[str], expected_clauses: list[str] = EXPECTED_CLAUSES
) -> str:
    missing = [clause for clause in expected_clauses if clause not in detected_clauses]
    if missing:
        return (
            "Missing standard clauses: "
            + ", ".join(missing)
            + ". Consider adding them for better protection."
        )
    return "All standard clauses are present."


def advice_on_unusual_combinations(detected_clauses: set[str]) -> str:
    if "TERMINATION" in detected_clauses and "NOTICE_PERIOD" not in detected_clauses:
        return "TERMINATION exists but NOTICE_PERIOD is missing. This may weaken protection."
    return ""


def get_clause_risk(clause_type: str, risk_df: pd.DataFrame) -> str:
    if risk_df.empty:
        return "unknown"
    match = risk_df[risk_df["clause_type"].str.lower() == clause_type.lower()]
    if not match.empty:
        return str(match["risk_level"].iloc[0])
    return "unknown"


def get_top_lawyers(lawyer_rag: list[dict], area: str = "Civil", top_n: int = 3) -> list[dict]:
    filtered = [lawyer for lawyer in lawyer_rag if lawyer.get("specialization", "").lower() == area.lower()]
    filtered.sort(key=lambda lawyer: -lawyer.get("experience", 0))
    return filtered[:top_n]


st.set_page_config(page_title="Legal Document Assistant", layout="wide")
st.title("Legal Document Q&A Assistant")
st.markdown("Upload a legal PDF, build embeddings, then summarize or ask questions about it.")

civil_rag = load_json_list(RAG_PATH, key="civil_law_data")
lawyer_rag = load_json_list(LAWYER_PATH)
risk_df = load_risk_data()

with st.sidebar:
    st.header("Status")
    if "vectors" in st.session_state:
        st.success("Vector DB ready")
    else:
        st.warning("Vector DB not initialized")

    if "embedding_backend" in st.session_state:
        st.caption(f"Embeddings: {st.session_state.embedding_backend}")
    if "groq_model" in st.session_state:
        st.caption(f"LLM: {st.session_state.groq_model}")

    missing_keys = []
    if not GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")

    if missing_keys:
        st.info("Set these before using summaries and Q&A: " + ", ".join(missing_keys))
    elif GOOGLE_API_KEY:
        st.success("API keys detected")
    else:
        st.info("GROQ key detected. Embeddings will use the built-in local fallback.")

uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])
file_path = None

if uploaded_file:
    UPLOAD_DIR.mkdir(exist_ok=True)
    file_path = UPLOAD_DIR / uploaded_file.name
    with file_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())
    st.success("Document uploaded successfully.")
    st.caption(
        "Embeddings use larger chunks now to stay within Google's free-tier quota whenever possible."
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
            # Prefer only the uploaded PDF chunks for summarisation
            all_docs: list[Document] = st.session_state.final_docs
            pdf_docs = [
                doc for doc in all_docs if doc.metadata.get("source") == "PDF"
            ] or all_docs

            summary_docs = build_summary_docs(pdf_docs)

            response = invoke_with_groq_fallback(
                lambda llm: create_stuff_documents_chain(llm, summary_prompt).invoke(
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
                    create_stuff_documents_chain(llm, qa_prompt),
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

st.markdown(
    """
---
Disclaimer: This tool provides simplified explanations of legal documents.
It is not a substitute for professional legal advice.
"""
)
