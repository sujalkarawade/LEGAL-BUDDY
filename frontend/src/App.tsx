import { useEffect, useState } from "react";
import { Routes, Route, Navigate, useNavigate } from "react-router-dom";
import { embedDocument, getAnalysis, getStatus, summarize, uploadDocument } from "./api";
import Sidebar from "./components/Sidebar";
import "./App.css";

import UploadPage from "./pages/UploadPage";
import SummaryPage from "./pages/SummaryPage";
import ClausesPage from "./pages/ClausesPage";
import QAPage from "./pages/QAPage";

interface Status {
  groq: boolean;
  openrouter: boolean;
}

interface Analysis {
  // Define analysis structure based on your API response
  [key: string]: any;
}

interface UploadPageProps {
  filename: string | null;
  embedded: boolean;
  loading: boolean;
  error: string;
  handleUpload: (file: File) => void;
  handleEmbed: () => void;
}

interface SummaryPageProps {
  summary: string;
  loading: boolean;
  embedded: boolean;
  error: string;
  handleSummarize: () => void;
}

interface ClausesPageProps {
  analysis: Analysis | null;
  loading: boolean;
  embedded: boolean;
  error: string;
  handleAnalyze: () => void;
}

interface QAPageProps {
  embedded: boolean;
}

export default function App() {
  const [status, setStatus] = useState<Status>({ groq: false, openrouter: false });
  const [filename, setFilename] = useState<string | null>(null);
  const [embedded, setEmbedded] = useState<boolean>(false);
  const [embedBackend, setEmbedBackend] = useState<string>("");
  const [summary, setSummary] = useState<string>("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  
  // App level loading state
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const navigate = useNavigate();

  useEffect(() => {
    getStatus().then(setStatus).catch(() => {});
  }, []);

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
      setEmbedBackend(data.backend);
      // Auto redirect to summary after embed
      navigate('/summary');
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
      <Sidebar status={status} embedded={embedded} embedBackend={embedBackend} />
      
      <main className="main-content">
        <div className="page-content">
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

        <footer className="app-footer">
          Disclaimer: This tool provides AI-assisted insights and is not a substitute for professional legal advice.
        </footer>
      </main>
    </div>
  );
}
