import { useEffect, useState } from "react";
import { embedDocument, getAnalysis, getStatus, summarize, uploadDocument } from "./api";
import AnalysisPanel from "./components/AnalysisPanel";
import QAPanel from "./components/QAPanel";
import Sidebar from "./components/Sidebar";
import "./index.css";

export default function App() {
  const [status, setStatus] = useState({ groq: false, openrouter: false });
  const [filename, setFilename] = useState(null);
  const [embedded, setEmbedded] = useState(false);
  const [embedBackend, setEmbedBackend] = useState("");
  const [summary, setSummary] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    getStatus().then(setStatus).catch(() => {});
  }, []);

  async function handleUpload(file) {
    setError("");
    setLoading("Uploading...");
    try {
      const data = await uploadDocument(file);
      setFilename(data.filename);
      setEmbedded(false);
      setSummary("");
      setAnalysis(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading("");
    }
  }

  async function handleEmbed() {
    if (!filename) return;
    setError("");
    setLoading("Building embeddings...");
    try {
      const data = await embedDocument(filename);
      setEmbedded(true);
      setEmbedBackend(data.backend);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading("");
    }
  }

  async function handleSummarize() {
    setError("");
    setLoading("Summarizing...");
    try {
      const data = await summarize();
      setSummary(data.summary);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading("");
    }
  }

  async function handleAnalyze() {
    setError("");
    setLoading("Analyzing clauses...");
    try {
      const data = await getAnalysis();
      setAnalysis(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading("");
    }
  }

  return (
    <div className="layout">
      <Sidebar status={status} embedded={embedded} embedBackend={embedBackend} />
      <main className="main">
        <h1>Legal Document Q&amp;A Assistant</h1>
        <p className="subtitle">Upload a legal PDF, build embeddings, then summarize or ask questions about it.</p>

        <section className="card">
          <label className="upload-label">
            <input type="file" accept=".pdf" onChange={(e) => e.target.files[0] && handleUpload(e.target.files[0])} />
            <span>Choose PDF</span>
          </label>
          {filename && <span className="filename">{filename}</span>}
        </section>

        {filename && (
          <section className="card actions">
            <button onClick={handleEmbed} disabled={!!loading}>Embed Document</button>
            <button onClick={handleSummarize} disabled={!embedded || !!loading}>Summarize</button>
            <button onClick={handleAnalyze} disabled={!embedded || !!loading}>Analyze Clauses</button>
          </section>
        )}

        {loading && <p className="info">{loading}</p>}
        {error && <p className="error">{error}</p>}

        {summary && (
          <section className="card">
            <h2>Summary</h2>
            <p>{summary}</p>
          </section>
        )}

        {embedded && <QAPanel />}
        {analysis && <AnalysisPanel data={analysis} />}

        <footer className="disclaimer">
          Disclaimer: This tool provides simplified explanations of legal documents. It is not a substitute for professional legal advice.
        </footer>
      </main>
    </div>
  );
}
