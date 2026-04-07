import { useEffect, useState } from "react";
import { Routes, Route, Navigate, useNavigate } from "react-router-dom";
import { embedDocument, getAnalysis, getStatus, summarize, uploadDocument } from "./api";
import Sidebar from "./components/Sidebar";

import UploadPage from "./pages/UploadPage";
import SummaryPage from "./pages/SummaryPage";
import ClausesPage from "./pages/ClausesPage";
import QAPage from "./pages/QAPage";

export default function App() {
  const [status, setStatus] = useState({ groq: false, openrouter: false });
  const [filename, setFilename] = useState(null);
  const [embedded, setEmbedded] = useState(false);
  const [embedBackend, setEmbedBackend] = useState("");
  const [summary, setSummary] = useState("");
  const [analysis, setAnalysis] = useState(null);
  
  // App level loading state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const navigate = useNavigate();

  useEffect(() => {
    getStatus().then(setStatus).catch(() => {});
  }, []);

  async function handleUpload(file) {
    if (!file) return;
    setError("");
    setLoading(true);
    try {
      const data = await uploadDocument(file);
      setFilename(data.filename);
      setEmbedded(false);
      setSummary("");
      setAnalysis(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleEmbed() {
    if (!filename) return;
    setError("");
    setLoading(true);
    try {
      const data = await embedDocument(filename);
      setEmbedded(true);
      setEmbedBackend(data.backend);
      // Auto redirect to summary after embed
      navigate('/summary');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleSummarize() {
    setError("");
    setLoading(true);
    try {
      const data = await summarize();
      setSummary(data.summary);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleAnalyze() {
    setError("");
    setLoading(true);
    try {
      const data = await getAnalysis();
      setAnalysis(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-screen overflow-hidden bg-black text-gray-100 selection:bg-accent-blue/30 selection:text-white relative">
      <Sidebar status={status} embedded={embedded} embedBackend={embedBackend} />
      
      <main className="flex-1 overflow-y-auto flex flex-col relative">
        <div className="p-8 flex-1">
          <Routes>
            <Route path="/" element={<Navigate to="/upload" replace />} />
            
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
            
            <Route 
              path="/qa" 
              element={<QAPage embedded={embedded} />} 
            />
          </Routes>
        </div>

        <footer className="w-full mt-auto p-4 border-t border-white/[0.06] bg-black/90 backdrop-blur-sm text-center text-xs text-gray-600">
          Disclaimer: This tool provides AI-assisted insights and is not a substitute for professional legal advice.
        </footer>
      </main>
    </div>
  );
}

