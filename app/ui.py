import streamlit as st

from app.config import DISCLAIMER_PATH, GROQ_API_KEY, OPENROUTER_API_KEY, STYLES_PATH


def inject_global_styles() -> None:
    if STYLES_PATH.exists():
        st.markdown(f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_page_intro() -> None:
    st.title("Legal Document Q&A Assistant")
    st.markdown("Upload a legal PDF, build embeddings, then summarize or ask questions about it.")


def render_sidebar_status() -> None:
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

        if GROQ_API_KEY:
            st.success("Groq API detected")
        if OPENROUTER_API_KEY:
            st.success("OpenRouter API detected")

        if missing_keys:
            st.info("Set these before using summaries and Q&A: " + ", ".join(missing_keys))
        elif GROQ_API_KEY and not OPENROUTER_API_KEY:
            st.info("GROQ key detected. Embeddings will use the built-in local fallback.")


def render_footer() -> None:
    if DISCLAIMER_PATH.exists():
        st.markdown(DISCLAIMER_PATH.read_text(encoding="utf-8"), unsafe_allow_html=True)

