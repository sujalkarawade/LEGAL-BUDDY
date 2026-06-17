import { useState } from "react";
import { Routes, Route, Navigate, useNavigate } from "react-router-dom";
import { embedDocument, getAnalysis, summarize, uploadDocument, AnalysisResponse } from "./api";
import "./App.css";

import Dashboard from "./pages/Dashboard";
import UploadPage from "./pages/UploadPage";
import SummaryPage from "./pages/SummaryPage";
import ClausesPage from "./pages/ClausesPage";
import QAPage from "./pages/QAPage";
import GenerateDocumentPage from "./pages/GenerateDocumentPage";

export default function App() {
  const [filename, setFilename]   = useState<string | null>(null);
  const [embedded, setEmbedded]   = useState<boolean>(false);
  const [summary, setSummary]     = useState<string>("");
  const [analysis, setAnalysis]   = useState<AnalysisResponse | null>(null);
  const [loading, setLoading]     = useState<boolean>(false);
  const [error, setError]         = useState<string>("");

  const navigate = useNavigate();

  async function handleUpload(file: File): Promise<void> {
    if (!file) return;
    setError("");
    setLoading(true);
    try {
      const data = await uploadDocument(file);
      setFilename(data.filename);
      setEmbedded(false);
      setSummary("");
      setAnalysis(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleEmbed(): Promise<void> {
    if (!filename) return;
    setError("");
    setLoading(true);
    try {
      const data = await embedDocument(filename);
      setEmbedded(true);
      void data;
      navigate("/summary");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleSummarize(): Promise<void> {
    setError("");
    setLoading(true);
    try {
      const data = await summarize();
      setSummary(data.summary);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleAnalyze(): Promise<void> {
    setError("");
    setLoading(true);
    try {
      const data = await getAnalysis();
      setAnalysis(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-container">
      <Routes>
        <Route path="/" element={<Dashboard />} />

        <Route
          path="/upload"
          element={
            <UploadPage
              filename={filename}
              embedded={embedded}
              loading={loading}
              error={error}
              handleUpload={handleUpload}
              handleEmbed={handleEmbed}
            />
          }
        />

        <Route
          path="/summary"
          element={
            <SummaryPage
              summary={summary}
              loading={loading}
              embedded={embedded}
              error={error}
              handleSummarize={handleSummarize}
            />
          }
        />

        <Route
          path="/clauses"
          element={
            <ClausesPage
              analysis={analysis}
              loading={loading}
              embedded={embedded}
              error={error}
              handleAnalyze={handleAnalyze}
            />
          }
        />

        <Route path="/qa" element={<QAPage embedded={embedded} />} />

        <Route path="/generate" element={<GenerateDocumentPage />} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
