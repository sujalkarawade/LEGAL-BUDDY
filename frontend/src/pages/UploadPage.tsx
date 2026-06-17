import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, File, Activity, Loader2, CheckCircle2, FileText, ArrowLeft } from "lucide-react";
import "./UploadPage.css";

interface UploadPageProps {
  filename: string | null;
  embedded: boolean;
  loading: boolean;
  error: string;
  handleUpload: (file: File) => void;
  handleEmbed: () => void;
}

export default function UploadPage({
  filename,
  embedded,
  loading,
  error,
  handleUpload,
  handleEmbed,
}: UploadPageProps) {
  const [isDragging, setIsDragging] = useState(false);
  const navigate = useNavigate();

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) {
      handleUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="upload-shell">
      {/* Top Bar */}
      <div className="upload-topbar">
        <button
          id="btn-back-upload"
          className="back-btn"
          onClick={() => navigate("/")}
        >
          <ArrowLeft size={15} /> Dashboard
        </button>
        <div className="topbar-brand">
          <FileText size={16} style={{ color: "#e8b84b" }} />
          Upload Legal Document
        </div>
        <div style={{ width: 110 }} />
      </div>

      {/* Page Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="upload-page"
      >
        <div className="upload-header">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.1, duration: 0.4 }}
            className="upload-icon-wrapper"
          >
            <FileText className="header-icon" size={28} />
          </motion.div>
          <h1 className="upload-title">Upload Legal Document</h1>
          <p className="upload-description">
            Upload a PDF contract or legal file to securely build embeddings and begin AI-assisted analysis.
          </p>
        </div>

        <motion.div
          className={`upload-area ${isDragging ? "dragging" : ""} ${filename ? "has-file" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          whileHover={!filename ? { scale: 1.01 } : {}}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
        >
          <input
            type="file"
            accept=".pdf"
            id="file-upload"
            className="file-input"
            onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
          />

          <label htmlFor="file-upload" className="upload-label">
            <AnimatePresence mode="wait">
              {isDragging ? (
                <motion.div
                  key="dragging"
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.8, opacity: 0 }}
                  className="upload-icon dragging-icon"
                >
                  <UploadCloud size={40} />
                </motion.div>
              ) : (
                <motion.div
                  key="normal"
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.8, opacity: 0 }}
                  className="upload-icon"
                >
                  <UploadCloud size={36} />
                </motion.div>
              )}
            </AnimatePresence>

            <div className="upload-text-content">
              <p className="upload-text">Click to upload or drag and drop</p>
              <p className="upload-subtext">Supported format: PDF only</p>
            </div>
          </label>

          <div className="blob blob-1" />
          <div className="blob blob-2" />
        </motion.div>

        <AnimatePresence>
          {filename && (
            <motion.div
              initial={{ opacity: 0, y: 15, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ type: "spring", bounce: 0.4 }}
              className="file-info"
            >
              <div className="file-details">
                <div className="file-icon-container">
                  <File size={22} className="file-icon-svg" />
                </div>
                <div className="file-text-wrapper">
                  <p className="file-name">{filename}</p>
                  <p className="file-status">
                    {embedded ? "Successfully embedded and ready" : "Ready for embedding"}
                  </p>
                </div>
              </div>

              <button
                id="btn-embed-document"
                onClick={handleEmbed}
                disabled={loading || embedded}
                className={`embed-button ${embedded ? "embed-button-success" : "embed-button-primary"}`}
              >
                {loading ? (
                  <><Loader2 size={18} className="animate-spin" /> Processing...</>
                ) : embedded ? (
                  <><CheckCircle2 size={18} /> Embedded</>
                ) : (
                  <><Activity size={18} /> Embed Document</>
                )}
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="error-message"
            >
              <div className="error-icon">!</div>
              <p>{error}</p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}
