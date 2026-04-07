import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import nest_asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
nest_asyncio.apply()

from backend.routers import analysis, documents, qa  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Legal Document Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(qa.router, prefix="/api/qa", tags=["qa"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])


@app.get("/api/status")
def status():
    groq_key = bool(os.getenv("GROQ_API_KEY", "").strip())
    openrouter_key = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    return {"groq": groq_key, "openrouter": openrouter_key}
