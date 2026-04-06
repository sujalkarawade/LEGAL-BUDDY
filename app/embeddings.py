import hashlib
import json
import re
import urllib.error
import urllib.request

import numpy as np
import streamlit as st
from langchain_core.embeddings import Embeddings

from app.config import (
    LOCAL_EMBEDDING_DIMENSION,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBEDDING_MODEL,
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


class OpenRouterEmbeddings(Embeddings):
    """OpenRouter embeddings via its OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = OPENROUTER_BASE_URL,
        app_name: str = "Legal Buddy",
        site_url: str = "http://localhost:8501",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.site_url = site_url

    def _fetch_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            reason = detail or exc.reason
            raise RuntimeError(f"OpenRouter embeddings request failed ({exc.code}): {reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter embeddings request failed: {exc.reason}") from exc

        data = body.get("data", [])
        if not data:
            raise RuntimeError("OpenRouter embeddings response did not include any vectors.")

        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fetch_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        embeddings = self._fetch_embeddings([text])
        return embeddings[0] if embeddings else []


@st.cache_resource(show_spinner=False)
def build_openrouter_embeddings() -> OpenRouterEmbeddings:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Set it in the environment or Streamlit secrets.")
    return OpenRouterEmbeddings(
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_EMBEDDING_MODEL,
    )


@st.cache_resource(show_spinner=False)
def build_local_embeddings() -> LocalHashedEmbeddings:
    return LocalHashedEmbeddings()


def summarize_embedding_failure(exc: Exception) -> str:
    message = str(exc)
    if "429" in message or "rate limit" in message.lower():
        return "OpenRouter embedding quota or rate limit was reached"
    if "403" in message or "401" in message:
        return "OpenRouter rejected the embedding request"
    if "404" in message or "not found" in message.lower():
        return "The selected OpenRouter embedding model is unavailable"
    return "OpenRouter embeddings are unavailable right now"

