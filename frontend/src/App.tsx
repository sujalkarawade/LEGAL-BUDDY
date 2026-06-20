import { useState } from "react";
import { Routes, Route, Navigate, useNavigate } from "react-router-dom";
import { embedDocument, uploadDocument } from "./api";
import "./App.css";

import Dashboard from "./pages/Dashboard";
import UploadPage from "./pages/UploadPage";
import GenerateDocumentPage from "./pages/GenerateDocumentPage";
import DocumentIntelligencePage from "./pages/DocumentIntelligencePage";

export default function App() {
  const [filename, setFilename]   = useState<string | null>(null);
  const [embedded, setEmbedded]   = useState<boolean>(false);
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
      navigate("/analysis");
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

        <Route path="/generate" element={<GenerateDocumentPage />} />

        <Route
          path="/analysis"
          element={
            <DocumentIntelligencePage
              filename={filename}
              embedded={embedded}
            />
          }
        />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
