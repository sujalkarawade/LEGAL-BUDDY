# Legal Buddy

Legal Buddy is a Streamlit app for uploading legal PDFs, building embeddings, summarizing documents, answering questions, and flagging potentially risky clauses.

## Project structure

```text
LEGAL BUDDY/
|-- main.py
|-- app/
|   |-- config.py
|   |-- data.py
|   |-- embeddings.py
|   |-- llm.py
|   |-- analysis.py
|   |-- document_processing.py
|   |-- prompts.py
|   |-- ui.py
|-- assets/
|   |-- styles.css
|   |-- disclaimer.html
|-- requirements.txt
|-- .env
```

## Run locally

1. Open PowerShell in `d:\DON\LEGAL BUDDY`.
2. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run this once for the current shell and try again:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Create or update `.env` in the project root:

```env
GROQ_API_KEY=your-groq-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

5. Start the app:

```powershell
streamlit run main.py
```

6. Open `http://localhost:8501`.

## API key behavior

- `GROQ_API_KEY` is required for summaries and question answering.
- `OPENROUTER_API_KEY` is optional and is used for embeddings when available.
- If `OPENROUTER_API_KEY` is missing or rate-limited, the app falls back to local embeddings.
- `OPENROUTER_EMBEDDING_MODEL` is optional. If omitted, the app uses `openai/text-embedding-3-small`.
- The app can read values from `.env`, environment variables, or Streamlit secrets.

## Notes

- Uploaded PDFs are stored in `uploaded_docs/`.
- The sidebar shows separate detection blocks for available API providers.
- `main.py` is now the entrypoint only; most logic lives inside the `app/` package.
