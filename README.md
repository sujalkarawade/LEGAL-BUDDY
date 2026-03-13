# Legal Buddy

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Set your API keys in PowerShell:

```powershell
$env:GROQ_API_KEY="your-groq-api-key"
$env:GOOGLE_API_KEY="your-google-api-key"
```

4. Start the app:

```powershell
python -m streamlit run main.py
```

The app opens at `http://localhost:8501`.

## Notes

- `GOOGLE_API_KEY` is used for embeddings.
- `GROQ_API_KEY` is used for summaries and question answering.
- The app now reads keys from environment variables or Streamlit secrets.
