import { motion } from "framer-motion";
import { UploadCloud, File, Activity, Loader2 } from "lucide-react";
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
  handleEmbed 
}: UploadPageProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="upload-page"
    >
      <div className="upload-header">
        <h1 className="upload-title">Upload Legal Document</h1>
        <p className="upload-description">Upload a PDF contract or legal file to build embeddings and begin analysis.</p>
      </div>

      <div className="upload-area">
        <input
          type="file"
          accept=".pdf"
          id="file-upload"
          className="file-input"
          onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
        />
        
        <label htmlFor="file-upload" className="upload-label">
          <div className="upload-icon">
            <UploadCloud size={32} />
          </div>
          <div>
            <p className="upload-text">Click to upload or drag and drop</p>
            <p className="upload-subtext">PDF documents only</p>
          </div>
        </label>
      </div>

      {filename && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="file-info"
        >
          <div className="file-details">
            <div className="file-icon">
              <File size={20} className="file-icon-svg" />
            </div>
            <div>
              <p className="file-name">{filename}</p>
              <p className="file-status">Ready to embed</p>
            </div>
          </div>
          
          <button
            onClick={handleEmbed}
            disabled={loading || embedded}
            className={`embed-button ${embedded ? 'embed-button-success' : 'embed-button-primary'} disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {loading ? (
              <><Loader2 size={16} className="animate-spin" /> Processing</>
            ) : embedded ? (
              "Embedded"
            ) : (
              <><Activity size={16} /> Embed Document</>
            )}
          </button>
        </motion.div>
      )}

      {error && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="error-message">
          {error}
        </motion.div>
      )}
    </motion.div>
  );
}
