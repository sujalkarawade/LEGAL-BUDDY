import { motion, AnimatePresence } from "framer-motion";
import { FileText, Loader2, Sparkles, ScrollText } from "lucide-react";
import ReactMarkdown from "react-markdown";
import "./SummaryPage.css";

interface SummaryPageProps {
  summary: string;
  handleSummarize: () => void;
  loading: boolean;
  embedded: boolean;
  error: string;
}

export default function SummaryPage({
  summary,
  handleSummarize,
  loading,
  embedded,
  error,
}: SummaryPageProps) {
  if (!embedded) {
    return (
      <div className="no-document-state">
        <div className="no-document-icon">
          <FileText size={32} />
        </div>
        <h2 className="no-document-title">No Document Embedded</h2>
        <p className="no-document-description">
          Upload and embed a document first to generate a summary.
        </p>
      </div>
    );
  }

  const hasSummary = !!summary;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className={`summary-page ${hasSummary ? "summary-page--has-content" : "summary-page--empty"}`}
    >
      <AnimatePresence mode="wait">
        {!hasSummary && !loading ? (
          /* ── Hero empty state ── */
          <motion.div
            key="hero"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.35 }}
            className="summary-hero"
          >
            {/* Decorative dot grid */}
            <div className="hero-dots">
              {Array.from({ length: 16 }).map((_, i) => (
                <div key={i} className="hero-dot" />
              ))}
            </div>

            <div className="hero-icon-ring">
              <div className="hero-icon-inner">
                <ScrollText size={26} />
              </div>
            </div>

            <div className="hero-badge">
              <span className="hero-badge-dot" />
              AI Powered
            </div>

            <h1 className="summary-title">Document Summary</h1>
            <p className="summary-description">
              Let AI extract and summarize the key points, obligations, and clauses from your legal document.
            </p>

            <button
              onClick={handleSummarize}
              disabled={loading}
              className="generate-button disabled:opacity-50"
            >
              <Sparkles size={16} />
              Generate Summary
            </button>
          </motion.div>
        ) : loading ? (
          /* ── Loading state ── */
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="summary-hero"
          >
            <div className="hero-icon-ring">
              <div className="hero-icon-inner">
                <Loader2 size={26} className="animate-spin" />
              </div>
            </div>
            <h1 className="summary-title">Generating Summary</h1>
            <p className="summary-description">Analyzing your document and extracting key insights…</p>
          </motion.div>
        ) : (
          /* ── Content state ── */
          <motion.div
            key="content"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
            style={{ width: "100%" }}
          >
            {/* Header row */}
            <div className="summary-header">
              <div className="summary-header-text">
                <h1 className="summary-title">Document Summary</h1>
                <p className="summary-description">AI-generated complete summary of the key points.</p>
              </div>
              <button
                onClick={handleSummarize}
                disabled={loading}
                className="generate-button disabled:opacity-50"
              >
                <Sparkles size={14} />
                Regenerate
              </button>
            </div>

            {error && <div className="error-message">{error}</div>}

            {/* Content card */}
            <div className="summary-content">
              <div className="summary-content-inner">
                <div className="summary-text">
                  <ReactMarkdown>{summary}</ReactMarkdown>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
