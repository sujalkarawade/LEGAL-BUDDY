import streamlit as st
from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, GROQ_MODEL_CANDIDATES


@st.cache_resource(show_spinner=False)
def build_llm(model_name: str) -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY. Set it in the environment or Streamlit secrets.")
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)


def groq_model_candidates() -> list[str]:
    seen = set()
    candidates = []
    for model_name in GROQ_MODEL_CANDIDATES:
        if model_name and model_name not in seen:
            seen.add(model_name)
            candidates.append(model_name)
    return candidates


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

