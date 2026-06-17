import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { UploadCloud, FileText, Sparkles, ShieldCheck, Zap, Brain, ArrowRight, Scale } from "lucide-react";
import "./Dashboard.css";

const fadeUp = {
  hidden: { opacity: 0, y: 28 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.12, duration: 0.55, ease: "easeOut" as const },
  }),
};

export default function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="dashboard">
      {/* Ambient Orbs */}
      <div className="dashboard-orb dashboard-orb-1" />
      <div className="dashboard-orb dashboard-orb-2" />
      <div className="dashboard-orb dashboard-orb-3" />

      <div className="dashboard-content">
        {/* ── Hero Header ── */}
        <motion.div
          className="dashboard-hero"
          initial="hidden"
          animate="visible"
        >
          <motion.div custom={0} variants={fadeUp} className="dashboard-badge">
            <span className="badge-dot" />
            AI-Powered Legal Assistant
          </motion.div>

          <motion.h1 custom={1} variants={fadeUp} className="dashboard-title">
            Legal Intelligence,{" "}
            <span className="title-highlight">Simplified.</span>
          </motion.h1>

          <motion.p custom={2} variants={fadeUp} className="dashboard-subtitle">
            Upload your legal documents for instant AI analysis, or generate
            professional legal documents from scratch — tailored to your exact needs.
          </motion.p>
        </motion.div>

        {/* ── Action Cards ── */}
        <motion.div
          className="dashboard-cards"
          initial="hidden"
          animate="visible"
        >
          {/* Upload Card */}
          <motion.div
            custom={3}
            variants={fadeUp}
            className="action-card card-upload"
            whileHover={{ y: -4, transition: { type: "spring", stiffness: 300, damping: 22 } }}
            onClick={() => navigate("/upload")}
          >
            <div className="card-corner-glow glow-upload" />
            <div className="card-inner">
              <div className="card-icon-wrap icon-upload">
                <UploadCloud size={26} />
              </div>

              <div className="card-text">
                <h2 className="card-title">Upload a Document</h2>
                <p className="card-desc">
                  Analyze existing contracts, agreements, or legal files with
                  AI-powered clause detection, risk scoring, and Q&amp;A.
                </p>
              </div>

              <div className="card-features">
                <div className="card-feature">
                  <span className="feature-dot dot-upload" />
                  <ShieldCheck size={13} />
                  Clause risk analysis
                </div>
                <div className="card-feature">
                  <span className="feature-dot dot-upload" />
                  <Brain size={13} />
                  AI-powered summarization
                </div>
                <div className="card-feature">
                  <span className="feature-dot dot-upload" />
                  <Zap size={13} />
                  Ask questions about the document
                </div>
              </div>

              <button
                id="btn-upload-document"
                className="card-cta cta-upload"
                onClick={(e) => { e.stopPropagation(); navigate("/upload"); }}
              >
                Upload Document <ArrowRight size={16} />
              </button>
            </div>
          </motion.div>

          {/* Generate Card */}
          <motion.div
            custom={4}
            variants={fadeUp}
            className="action-card card-generate"
            whileHover={{ y: -4, transition: { type: "spring", stiffness: 300, damping: 22 } }}
            onClick={() => navigate("/generate")}
          >
            <div className="card-corner-glow glow-generate" />
            <div className="card-inner">
              <div className="card-icon-wrap icon-generate">
                <FileText size={26} />
              </div>

              <div className="card-text">
                <h2 className="card-title">Generate a Document</h2>
                <p className="card-desc">
                  Create professional legal documents from scratch — NDAs,
                  contracts, agreements — tailored to your context and jurisdiction.
                </p>
              </div>

              <div className="card-features">
                <div className="card-feature">
                  <span className="feature-dot dot-generate" />
                  <Scale size={13} />
                  9+ document types
                </div>
                <div className="card-feature">
                  <span className="feature-dot dot-generate" />
                  <Sparkles size={13} />
                  AI-generated, legally structured
                </div>
                <div className="card-feature">
                  <span className="feature-dot dot-generate" />
                  <Zap size={13} />
                  Copy or download instantly
                </div>
              </div>

              <button
                id="btn-generate-document"
                className="card-cta cta-generate"
                onClick={(e) => { e.stopPropagation(); navigate("/generate"); }}
              >
                Generate Document <ArrowRight size={16} />
              </button>
            </div>
          </motion.div>
        </motion.div>

        {/* ── Footer Note ── */}
        <motion.p
          custom={5}
          variants={fadeUp}
          initial="hidden"
          animate="visible"
          className="dashboard-footer-note"
        >
          AI-assisted insights · Not a substitute for professional legal advice
        </motion.p>
      </div>
    </div>
  );
}
