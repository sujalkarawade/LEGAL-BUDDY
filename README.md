# Legal Buddy

Legal Buddy is a legal document analysis tool. Upload a PDF, build embeddings, summarize the document, ask questions via RAG, and flag risky clauses — powered by a FastAPI backend and a Vite + React frontend.

## Project structure

```text
LEGAL BUDDY/
├── backend/
│   ├── main.py          # FastAPI app + CORS
│   ├── state.py         # In-memory session state
│   └── routers/
│       ├── documents.py # Upload & embed endpoints
│       ├── qa.py        # Summarize & ask endpoints
│       └── analysis.py  # Clause analysis endpoint
├── app/
│   ├── config.py
│   ├── data.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── analysis.py
│   ├── document_processing.py
│   └── prompts.py
├── frontend/            # Vite + React UI
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── components/
│           ├── Sidebar.jsx
│           ├── QAPanel.jsx
│           └── AnalysisPanel.jsx
├── assets/
├── requirements.txt
└── .env
```

## Setup

### 1. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```powershell
cd frontend
npm install
```

### 4. Configure environment variables

Create or update `.env` in the project root:

```env
GROQ_API_KEY=your-groq-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

## Running locally

Start both servers — each in its own terminal.

```powershell
# Terminal 1 — backend (from project root)
uvicorn backend.main:app --reload
```

```powershell
# Terminal 2 — frontend
cd frontend
npm run dev
```

Then open `http://localhost:5173`.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | API key detection |
| POST | `/api/documents/upload` | Upload a PDF |
| POST | `/api/documents/embed` | Build vector store |
| POST | `/api/qa/summarize` | Summarize document |
| POST | `/api/qa/ask` | Ask a question (RAG) |
| GET | `/api/analysis/clauses` | Clause analysis & risks |

## API key behavior

- `GROQ_API_KEY` — required for summaries and Q&A.
- `OPENROUTER_API_KEY` — optional, used for embeddings. Falls back to local embeddings if missing or rate-limited.
- `OPENROUTER_EMBEDDING_MODEL` — optional, defaults to `openai/text-embedding-3-small`.

## Notes

- Uploaded PDFs are saved to `uploaded_docs/`.
- Session state (vector store, docs) is held in memory for the lifetime of the backend process.
- The `app/` package contains all core logic and has no Streamlit dependencies.
