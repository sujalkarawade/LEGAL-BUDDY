import asyncio
import os
from pathlib import Path

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / "uploaded_docs"
RAG_PATH = BASE_DIR / "civil_law(RAG).json"
LAWYER_PATH = BASE_DIR / "lawyer(RAG).json"
RISK_PATH = BASE_DIR / "legal_contract_clauses.csv"
STYLES_PATH = BASE_DIR / "assets" / "styles.css"
DISCLAIMER_PATH = BASE_DIR / "assets" / "disclaimer.html"

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODEL_CANDIDATES = [
    os.getenv("GROQ_MODEL", "").strip(),
    DEFAULT_GROQ_MODEL,
    "llama-3.3-70b-versatile",
]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"
PDF_CHUNK_SIZE = 4000
PDF_CHUNK_OVERLAP = 400
FREE_TIER_EMBED_ITEM_BUDGET = 95
LOCAL_EMBEDDING_DIMENSION = 512


def initialize_runtime() -> None:
    load_dotenv(BASE_DIR / ".env")

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    nest_asyncio.apply()

    if GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    if OPENROUTER_API_KEY:
        os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY


def get_config_value(name: str) -> str:
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        return os.getenv(name, "").strip()

GROQ_API_KEY = get_config_value("GROQ_API_KEY")
OPENROUTER_API_KEY = get_config_value("OPENROUTER_API_KEY")
OPENROUTER_EMBEDDING_MODEL = (
    get_config_value("OPENROUTER_EMBEDDING_MODEL") or DEFAULT_OPENROUTER_EMBEDDING_MODEL
)
