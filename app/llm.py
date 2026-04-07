from functools import lru_cache

from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, GROQ_MODEL_CANDIDATES

_groq_model_used: str = ""


@lru_cache(maxsize=8)
def build_llm(model_name: str) -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY.")
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)


def groq_model_candidates() -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    for m in GROQ_MODEL_CANDIDATES:
        if m and m not in seen:
            seen.add(m)
            candidates.append(m)
    return candidates


def invoke_with_groq_fallback(factory):
    last_error = None
    for model_name in groq_model_candidates():
        try:
            llm = build_llm(model_name)
            result = factory(llm)
            global _groq_model_used
            _groq_model_used = model_name
            return result
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if any(k in message for k in ("model_decommissioned", "decommissioned", "not supported")):
                continue
            if "404" in message and "model" in message:
                continue
            raise
    raise RuntimeError(
        "No working Groq model available. Tried: "
        + ", ".join(groq_model_candidates())
        + f". Last error: {last_error}"
    )
