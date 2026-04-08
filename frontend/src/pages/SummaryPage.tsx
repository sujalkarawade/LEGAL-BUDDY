import { motion } from "framer-motion";
import { FileText, Loader2, Sparkles } from "lucide-react";
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
  error 
}: SummaryPageProps) {
  if (!embedded) {
    return (
      <div className="no-document-state">
        <div className="no-document-icon">
          <FileText size={32} />
        </div>
        <h2 className="no-document-title">No Document Embedded</h2>
        <p className="no-document-description">Upload and embed a document first to generate a summary.</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="summary-page"
    >
      <div className="summary-header">
        <div>
          <h1 className="summary-title">Document Summary</h1>
          <p className="summary-description">AI-generated complete summary of the key points.</p>
        </div>
        <button
          onClick={handleSummarize}
          disabled={loading}
          className="generate-button disabled:opacity-50"
        >
          {loading ? (
            <><Loader2 size={16} className="animate-spin" /> Generating...</>
          ) : (
            <><Sparkles size={16} /> Generate Summary</>
          )}
        </button>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {summary ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="summary-content"
        >
          <div className="summary-text">
            <ReactMarkdown>{summary}</ReactMarkdown>
          </div>
        </motion.div>
      ) : (
        !loading && (
          <div className="empty-summary">
            <Sparkles size={32} className="empty-icon" />
            <p className="empty-text">Click generate to create an AI summary.</p>
          </div>
        )
      )}
    </motion.div>
  );
}
