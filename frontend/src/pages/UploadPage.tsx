import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  UploadCloud,
  File,
  Activity,
  Loader2,
  CheckCircle2,
  FileText,
  ArrowLeft,
  ShieldCheck,
  Cpu,
  AlertTriangle,
  Lock,
} from "lucide-react";
import "./UploadPage.css";

interface UploadPageProps {
  filename: string | null;
  embedded: boolean;
  loading: boolean;
  error: string;
  handleUpload: (file: File) => void;
  handleEmbed: () => void;
}

// Floating legal symbols config
const LEGAL_SYMBOLS = [
  { char: "§", top: "12%", left: "6%",  size: 22, opacity: 0.04 },
  { char: "⚖", top: "30%", left: "3%",  size: 28, opacity: 0.035 },
  { char: "§", top: "70%", left: "8%",  size: 18, opacity: 0.03 },
  { char: "⚖", top: "85%", left: "4%",  size: 22, opacity: 0.04 },
  { char: "§", top: "18%", right: "5%", size: 24, opacity: 0.04 },
  { char: "⚖", top: "55%", right: "3%", size: 20, opacity: 0.035 },
  { char: "§", top: "78%", right: "7%", size: 26, opacity: 0.03 },
];

const TRUST_PILLS = [
  { icon: <ShieldCheck size={13} />, label: "Secure Processing" },
  { icon: <Cpu size={13} />,         label: "AI Analysis" },
  { icon: <AlertTriangle size={13} />, label: "Risk Detection" },
  { icon: <Lock size={13} />,        label: "Privacy Protected" },
];

const fadeUp = {
  hidden: { opacity: 0, y: 28 },
  visible: (delay = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1], delay },
  }),
};

const scaleIn = {
  hidden: { opacity: 0, scale: 0.88 },
  visible: (delay = 0) => ({
    opacity: 1,
    scale: 1,
    transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1], delay },
  }),
};

export default function UploadPage({
  filename,
  embedded,
  loading,
  error,
  handleUpload,
  handleEmbed,
}: UploadPageProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const navigate = useNavigate();

  // Simulate progress on file selection
  useEffect(() => {
    if (filename && !embedded) {
      setUploadProgress(0);
      const timer = setTimeout(() => setUploadProgress(100), 80);
      return () => clearTimeout(timer);
    }
    if (embedded) setUploadProgress(100);
  }, [filename, embedded]);

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
      {/* ── Background ── */}
      <div className="upload-bg-layer" aria-hidden="true">
        <div className="upload-bg-grid" />
        <div className="upload-bg-radial" />
        <div className="upload-bg-vignette" />
        <div className="upload-bg-noise" />
      </div>

      {/* ── Floating Legal Symbols ── */}
      <div className="upload-legal-symbols" aria-hidden="true">
        {LEGAL_SYMBOLS.map((s, i) => (
          <span
            key={i}
            className="upload-symbol"
            style={{
              top: s.top,
              left: (s as any).left,
              right: (s as any).right,
              fontSize: s.size,
              opacity: s.opacity,
            }}
          >
            {s.char}
          </span>
        ))}
      </div>



      {/* ── Back Button ── */}
      <motion.button
        id="btn-back-upload"
        className="back-btn"
        onClick={() => navigate("/")}
        initial={{ opacity: 0, x: -12 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        whileHover={{ scale: 1.03 }}
        whileTap={{ scale: 0.97 }}
        style={{ position: 'absolute', top: '2rem', left: '2rem', zIndex: 20 }}
      >
        <ArrowLeft size={13} />
        Back to Dashboard
      </motion.button>

      {/* ── Page Content ── */}
      <motion.div
        className="upload-page"
        initial="hidden"
        animate="visible"
        variants={{ visible: { transition: { staggerChildren: 0.08 } } }}
      >
        {/* ── Hero ── */}
        <div className="upload-hero">


          <motion.h1 className="upload-title" variants={fadeUp} custom={0.05}>
            Upload Legal Document
          </motion.h1>

          <motion.p className="upload-description" variants={fadeUp} custom={0.1}>
            Transform contracts and legal documents into actionable AI&nbsp;insights.
          </motion.p>
        </div>

        {/* ── Upload Zone Wrapper ── */}
        <motion.div
          className="upload-zone-wrapper"
          variants={fadeUp}
          custom={0.15}
        >
          {/* Drop Area */}
          <motion.div
            className={`upload-area${isDragging ? " dragging" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            whileHover={!filename ? { y: -2 } : {}}
            transition={{ type: "spring", stiffness: 260, damping: 22 }}
          >
            <input
              type="file"
              accept=".pdf,.docx,.txt"
              id="file-upload"
              className="file-input"
              onChange={(e) =>
                e.target.files?.[0] && handleUpload(e.target.files[0])
              }
            />

            <label htmlFor="file-upload" className="upload-label">
              <AnimatePresence mode="wait">
                {isDragging ? (
                  /* ── Drag State ── */
                  <motion.div
                    key="dragging"
                    initial={{ scale: 0.85, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.85, opacity: 0 }}
                    transition={{ duration: 0.22 }}
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: "1rem",
                    }}
                  >
                    <div className="upload-zone-icon">
                      <div className="upload-zone-icon-glow" />
                      <UploadCloud
                        className="upload-zone-svg"
                        size={46}
                        style={{ color: "#D4AF37" }}
                      />
                    </div>
                    <div className="upload-zone-text">
                      <p className="upload-drag-text">Release to Upload</p>
                      <p className="upload-drag-sub">Drop it right here</p>
                    </div>
                  </motion.div>
                ) : filename ? (
                  /* ── Has File — show minimal prompt ── */
                  <motion.div
                    key="has-file"
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.9, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: "0.75rem",
                    }}
                  >
                    <div className="upload-zone-icon">
                      <div className="upload-zone-icon-glow" />
                      <UploadCloud className="upload-zone-svg" size={38} />
                    </div>
                    <div className="upload-zone-text">
                      <p className="upload-zone-heading" style={{ fontSize: 16, letterSpacing: "0.1em" }}>
                        Replace Document
                      </p>
                      <p className="upload-zone-sub">Drag & Drop or Browse Files</p>
                    </div>
                  </motion.div>
                ) : (
                  /* ── Idle State ── */
                  <motion.div
                    key="idle"
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.9, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: "1.25rem",
                    }}
                  >
                    <div className="upload-zone-icon">
                      <div className="upload-zone-icon-glow" />
                      <UploadCloud className="upload-zone-svg" size={44} />
                    </div>
                    <div className="upload-zone-text">
                      <p className="upload-zone-heading">Drop Your Document</p>
                      <p className="upload-zone-sub">Drag &amp; Drop or Browse Files</p>
                      <div className="upload-zone-formats">
                        {["PDF", "DOCX", "TXT"].map((f, i, arr) => (
                          <span key={f}>
                            <span className="upload-zone-format-pill">{f}</span>
                            {i < arr.length - 1 && (
                              <span
                                className="upload-zone-format-sep"
                                style={{ display: "inline-block", width: 3, height: 3, borderRadius: "50%", background: "rgba(255,255,255,0.15)", margin: "0 4px", verticalAlign: "middle" }}
                              />
                            )}
                          </span>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </label>
          </motion.div>

          {/* ── File Card ── */}
          <AnimatePresence>
            {filename && (
              <motion.div
                className="file-card"
                initial={{ opacity: 0, y: 16, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.96 }}
                transition={{ type: "spring", bounce: 0.35, duration: 0.55 }}
              >
                <div className="file-card-left">
                  <div className="file-card-icon">
                    <File size={20} />
                  </div>
                  <div className="file-card-info">
                    <p className="file-name">{filename}</p>
                    <p
                      className={`file-status ${embedded ? "embedded" : "ready"}`}
                    >
                      {embedded
                        ? "✓ Embedded & ready for AI analysis"
                        : "Ready for embedding"}
                    </p>
                    <div className="file-progress-bar">
                      <motion.div
                        className="file-progress-fill"
                        initial={{ width: "0%" }}
                        animate={{ width: `${uploadProgress}%` }}
                        transition={{ duration: 0.6, ease: "easeOut" }}
                      />
                    </div>
                  </div>
                </div>

                <button
                  id="btn-embed-document"
                  onClick={handleEmbed}
                  disabled={loading || embedded}
                  className={`embed-button ${
                    embedded ? "embed-button-success" : "embed-button-primary"
                  }`}
                >
                  {loading ? (
                    <>
                      <Loader2 size={15} className="animate-spin" />
                      Processing…
                    </>
                  ) : embedded ? (
                    <>
                      <CheckCircle2 size={15} />
                      Embedded
                    </>
                  ) : (
                    <>
                      <Activity size={15} />
                      Embed Document
                    </>
                  )}
                </button>
              </motion.div>
            )}
          </AnimatePresence>

          {/* ── Trust Pills ── */}
          <motion.div
            className="upload-trust-pills"
            variants={fadeUp}
            custom={0.22}
          >
            {TRUST_PILLS.map((pill, i) => (
              <motion.div
                key={pill.label}
                className="trust-pill"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + i * 0.06, duration: 0.4, ease: "easeOut" }}
                whileHover={{ y: -1 }}
              >
                <span className="trust-pill-check">{pill.icon}</span>
                {pill.label}
              </motion.div>
            ))}
          </motion.div>
        </motion.div>

        {/* ── Error ── */}
        <AnimatePresence>
          {error && (
            <motion.div
              className="error-message"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.3 }}
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
