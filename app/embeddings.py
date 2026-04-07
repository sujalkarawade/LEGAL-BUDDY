import hashlib
import json
import re
import urllib.error
import urllib.request
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings

from app.config import (
    LOCAL_EMBEDDING_DIMENSION,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBEDDING_MODEL,
)


class LocalHashedEmbeddings(Embeddings):
    """Offline fallback embeddings."""

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
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


class OpenRouterEmbeddings(Embeddings):
    """OpenRouter embeddings via its OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str, base_url: str = OPENROUTER_BASE_URL,
                 app_name: str = "Legal Buddy", site_url: str = "http://localhost:8000"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.site_url = site_url

    def _fetch_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
            
        # OpenRouter has a bug where `input: ["single string"]` can return 404.
        # Passing `input: "single string"` or `input: ["two", "strings"]` works.
        input_data = texts[0] if len(texts) == 1 else texts
        payload = json.dumps({"model": self.model, "input": input_data}).encode("utf-8")
        
        req = urllib.request.Request(
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
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"OpenRouter embeddings failed ({exc.code}): {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter embeddings failed: {exc.reason}") from exc
            
        if "error" in body:
            error_data = body["error"]
            err_msg = error_data.get("message", error_data) if isinstance(error_data, dict) else error_data
            raise RuntimeError(f"OpenRouter failed: {err_msg}")
            
        data = body.get("data", [])
        if not data:
            raise RuntimeError("OpenRouter embeddings response had no vectors.")
        return [item["embedding"] for item in sorted(data, key=lambda x: x.get("index", 0))]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fetch_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        return (self._fetch_embeddings([text]) or [[]])[0]


@lru_cache(maxsize=1)
def build_openrouter_embeddings() -> OpenRouterEmbeddings:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY.")
    return OpenRouterEmbeddings(api_key=OPENROUTER_API_KEY, model=OPENROUTER_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
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
