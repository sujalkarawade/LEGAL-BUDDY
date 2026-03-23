# Legal Buddy

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Add your API keys to the `.env` file in the project root:

```env
GROQ_API_KEY=your-groq-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

4. Start the app:

```powershell
python -m streamlit run main.py
```

The app opens at `http://localhost:8501`.

## Notes

- `OPENROUTER_API_KEY` is used for embeddings.
- `GROQ_API_KEY` is used for summaries and question answering.
- `OPENROUTER_EMBEDDING_MODEL` is optional. If you omit it, the app uses `openai/text-embedding-3-small`.
- The app now reads keys from `.env`, environment variables, or Streamlit secrets.
