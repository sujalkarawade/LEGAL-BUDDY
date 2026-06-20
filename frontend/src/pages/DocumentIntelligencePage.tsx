import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { motion, useMotionValue, useSpring, type Transition } from "framer-motion";
import {
  ArrowLeft,
  FileText,
  BookOpen,
  List,
  AlertTriangle,
  ChevronRight,
  MessageSquare,
  Lightbulb,
  Shield,
  Sparkles,
  FileSearch,
  Gavel,
  Download,
  TrendingUp,
} from "lucide-react";
import "./DocumentIntelligencePage.css";

/* ─── Types ──────────────────────────────────────────────────── */
interface Props {
  filename: string | null;
  embedded: boolean;
}

/* ─── Helpers ────────────────────────────────────────────────── */
function detectDocType(name: string | null): string {
  if (!name) return "Legal Document";
  const lower = name.toLowerCase();
  if (lower.includes("nda") || lower.includes("non-disclosure")) return "NDA";
  if (lower.includes("rental") || lower.includes("lease")) return "Rental Agreement";
  if (lower.includes("employ") || lower.includes("offer")) return "Employment Contract";
  if (lower.includes("service")) return "Service Agreement";
  if (lower.includes("partner")) return "Partnership Agreement";
  if (lower.includes("sale") || lower.includes("purchase")) return "Sale Agreement";
  return "Legal Document";
}

/* ─── Animated Counter ───────────────────────────────────────── */
function Counter({ target, duration = 1200 }: { target: number; duration?: number }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let start = 0;
    const step = target / (duration / 16);
    const timer = setInterval(() => {
      start += step;
      if (start >= target) { setCount(target); clearInterval(timer); }
      else setCount(Math.floor(start));
    }, 16);
    return () => clearInterval(timer);
  }, [target, duration]);

  return <>{count}</>;
}

/* ─── Floating Particle ──────────────────────────────────────── */
function Particle({ x, y, delay }: { x: number; y: number; delay: number }) {
  return (
    <motion.div
      className="did-particle"
      style={{ left: `${x}%`, top: `${y}%` }}
      animate={{ y: [0, -28, 0], opacity: [0, 0.7, 0], scale: [0.5, 1.3, 0.5] }}
      transition={{ duration: 3.5 + Math.random() * 2, repeat: Infinity, delay, ease: "easeInOut" }}
    />
  );
}

/* ─── Risk Gauge ─────────────────────────────────────────────── */
const RISK_CONFIG = {
  low:    { score: 22, level: "Low",    cls: "low",    arc: 60  },
  medium: { score: 55, level: "Medium", cls: "medium", arc: 140 },
  high:   { score: 82, level: "High",   cls: "high",   arc: 220 },
};

function RiskGauge({ risk }: { risk: "low" | "medium" | "high" }) {
  const cfg = RISK_CONFIG[risk];
  const R = 52;
  const circ = 2 * Math.PI * R;
  // Arc covers ~240° of the circle; full = 240/360 * circ
  const fullArc = (240 / 360) * circ;
  const fillArc = (cfg.arc / 360) * circ;

  return (
    <div className="did-gauge-wrap">
      <svg width="140" height="120" className="did-gauge-svg" viewBox="0 0 140 120">
        {/* Track */}
        <circle
          className="did-gauge-track"
          cx="70" cy="80" r={R}
          strokeDasharray={`${fullArc} ${circ}`}
          strokeDashoffset={0}
          transform="rotate(-210 70 80)"
        />
        {/* Fill */}
        <motion.circle
          className={`did-gauge-fill did-gauge-fill--${cfg.cls}`}
          cx="70" cy="80" r={R}
          strokeDasharray={`0 ${circ}`}
          strokeDashoffset={0}
          transform="rotate(-210 70 80)"
          animate={{ strokeDasharray: `${fillArc} ${circ}` }}
          transition={{ duration: 1.4, ease: [0.22, 1, 0.36, 1], delay: 0.3 }}
        />
        <text x="70" y="76" className="did-gauge-center-value">{cfg.score}</text>
        <text x="70" y="90" className="did-gauge-center-label">RISK SCORE</text>
      </svg>
      <span className={`did-gauge-level did-gauge-level--${cfg.cls}`}>{cfg.level} Risk</span>
    </div>
  );
}

/* ─── Clause Chip ────────────────────────────────────────────── */
function ClauseChip({
  label, count, active, onClick,
}: { label: string; count: number; active: boolean; onClick: () => void }) {
  return (
    <motion.button
      className={`did-clause-chip ${active ? "did-clause-chip--active" : ""}`}
      onClick={onClick}
      whileHover={{ scale: 1.04 }}
      whileTap={{ scale: 0.97 }}
    >
      {label}
      <span className="did-clause-chip-count">{count}</span>
    </motion.button>
  );
}

/* ─── Static Demo Data ───────────────────────────────────────── */
const RISKS = [
  { title: "Unlimited Liability Exposure", desc: "No liability cap defined for consequential damages.", severity: "high" as const },
  { title: "Ambiguous Termination Clause", desc: "Termination window lacks clear notice period definition.", severity: "medium" as const },
  { title: "Jurisdiction Not Specified", desc: "Governing law clause references multiple jurisdictions.", severity: "low" as const },
];

const CLAUSES = [
  { label: "Termination", count: 4 },
  { label: "Liability", count: 7 },
  { label: "Payment Terms", count: 3 },
  { label: "Confidentiality", count: 5 },
  { label: "Governing Law", count: 2 },
  { label: "Indemnification", count: 3 },
  { label: "Force Majeure", count: 1 },
  { label: "IP Rights", count: 2 },
  { label: "Dispute Resolution", count: 2 },
  { label: "Non-Compete", count: 1 },
];

const RECOMMENDATIONS = [
  { text: "Add an arbitration clause to avoid costly litigation", tag: "missing" as const },
  { text: "Include a force majeure provision for unforeseen events", tag: "missing" as const },
  { text: "Clarify payment terms — current language is ambiguous", tag: "warning" as const },
  { text: "Define a clear liability cap (e.g. 12 months of fees)", tag: "review" as const },
  { text: "Specify exact jurisdiction and governing law", tag: "warning" as const },
];

const PROMPTS = [
  "Explain this contract in plain English",
  "What are the key risks?",
  "Summarize my obligations",
  "What penalties exist?",
  "Is this NDA enforceable?",
];

const SUMMARY_TEXT = `This agreement establishes a legally binding arrangement between the parties regarding the scope of services, compensation structure, and mutual obligations. The document contains standard clauses governing confidentiality, intellectual property rights, and dispute resolution mechanisms.

Key highlights include a 12-month term with automatic renewal provisions, specific performance milestones tied to payment schedules, and broad indemnification clauses that may warrant negotiation. The termination provisions allow either party to exit with 30 days written notice under certain conditions, though the definition of "material breach" remains somewhat ambiguous and should be reviewed carefully before execution.`;

/* ─── Card animation variants ────────────────────────────────── */
const cardVariants = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1, y: 0,
    transition: {
      delay: i * 0.08,
      duration: 0.55,
      ease: [0.22, 1, 0.36, 1],
    } as Transition,
  }),
};

const PARTICLES = Array.from({ length: 12 }, (_, i) => ({
  x: 5 + i * 8.5, y: 10 + (i % 4) * 20, delay: i * 0.4,
}));

/* ═══════════════════════════════════════════════════════════════
   MAIN COMPONENT
═══════════════════════════════════════════════════════════════ */
export default function DocumentIntelligencePage({ filename }: Props) {
  const navigate = useNavigate();
  const [activeClause, setActiveClause] = useState<string | null>("Liability");

  // Cursor spotlight
  const cursorX = useMotionValue(-500);
  const cursorY = useMotionValue(-500);
  const springX = useSpring(cursorX, { stiffness: 80, damping: 20 });
  const springY = useSpring(cursorY, { stiffness: 80, damping: 20 });

  const handleMouseMove = useCallback(
    (e: MouseEvent) => { cursorX.set(e.clientX); cursorY.set(e.clientY); },
    [cursorX, cursorY]
  );

  useEffect(() => {
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [handleMouseMove]);

  const docType = detectDocType(filename);
  const riskLevel: "low" | "medium" | "high" = "medium";
  const totalClauses = CLAUSES.reduce((s, c) => s + c.count, 0);

  const analyticsCards = [
    { icon: FileText, label: "Document Type", value: docType, sub: "Auto-detected", valueClass: "did-ac-value--gold" },
    { icon: BookOpen, label: "Total Pages", value: 14, sub: "Estimated", valueClass: "" },
    { icon: List, label: "Clauses Detected", value: totalClauses, sub: `${CLAUSES.length} types found`, valueClass: "" },
    { icon: AlertTriangle, label: "Risk Level", value: "MEDIUM", sub: "Score: 55/100", valueClass: "did-ac-value--medium" },
  ];

  return (
    <div className="did-shell">
      {/* ── Background ── */}
      <div className="did-bg" aria-hidden="true">
        <div className="did-bg-grid" />
        <div className="did-bg-radial-top" />
        <div className="did-bg-radial-bl" />
        <div className="did-bg-radial-tr" />
        <div className="did-bg-noise" />
        {PARTICLES.map((p, i) => <Particle key={i} {...p} />)}
      </div>

      {/* ── Cursor Spotlight ── */}
      <motion.div className="did-cursor-spotlight" style={{ x: springX, y: springY }} />

      {/* ── Main Content ── */}
      <div className="did-content">

        {/* ── Top Bar ── */}
        <div className="did-topbar">
          <motion.button
            className="did-back-btn"
            onClick={() => navigate("/upload")}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4 }}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
          >
            <ArrowLeft size={12} />
            Back to Upload
          </motion.button>

          <div className="did-topbar-right">
            <motion.div
              className="did-badge did-badge--status"
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2, duration: 0.4 }}
            >
              Analyzed
            </motion.div>
          </div>
        </div>

        {/* ── Page Header ── */}
        <motion.div
          className="did-header"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="did-header-badges">
            {filename && (
              <div className="did-badge did-badge--file" title={filename}>
                <FileText size={9} />
                {filename}
              </div>
            )}
            <div className="did-badge did-badge--type">
              <Gavel size={9} />
              {docType}
            </div>
          </div>

          <h1 className="did-title">
            Document <span>Intelligence</span> Dashboard
          </h1>
          <p className="did-subtitle">AI-powered analysis of your legal document</p>
        </motion.div>

        {/* ── Analytics Cards ── */}
        <div className="did-analytics-grid did-section-gap">
          {analyticsCards.map(({ icon: Icon, label, value, sub, valueClass }, i) => (
            <motion.div
              key={label}
              className="did-analytics-card"
              custom={i}
              initial="hidden"
              animate="visible"
              variants={cardVariants}
              whileHover={{ y: -3 }}
            >
              <div className="did-ac-icon"><Icon size={16} /></div>
              <div className="did-ac-label">{label}</div>
              <div className={`did-ac-value ${valueClass}`}>
                {typeof value === "number" ? <Counter target={value} duration={900} /> : value}
              </div>
              <div className="did-ac-sub">{sub}</div>
            </motion.div>
          ))}
        </div>

        {/* ── Main Grid ── */}
        <div className="did-main-grid did-section-gap">

          {/* ── Section 1: Executive Summary (full width) ── */}
          <motion.div
            className="did-card did-main-grid--wide"
            custom={4}
            initial="hidden"
            animate="visible"
            variants={cardVariants}
          >
            <div className="did-card-header">
              <div className="did-card-title-row">
                <div className="did-card-icon"><FileSearch size={14} /></div>
                <span className="did-card-title">Executive Summary</span>
              </div>
              <motion.button
                className="did-btn did-btn--ghost"
                onClick={() => navigate("/upload")}
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.97 }}
              >
                View Full Summary <ChevronRight size={12} />
              </motion.button>
            </div>

            <div style={{ display: "flex", gap: "2rem", alignItems: "flex-start" }}>
              <div className="did-summary-text" style={{ flex: 1 }}>
                {SUMMARY_TEXT}
              </div>
              <div className="did-summary-meta" style={{ flexShrink: 0, flexDirection: "column", gap: "1.25rem" }}>
                <div className="did-summary-stat">
                  <div className="did-summary-stat-value"><Counter target={98} />%</div>
                  <div className="did-summary-stat-label">AI Confidence</div>
                </div>
                <div className="did-summary-divider" style={{ width: "100%", height: "1px" }} />
                <div className="did-summary-stat">
                  <div className="did-summary-stat-value"><Counter target={2} />m</div>
                  <div className="did-summary-stat-label">Read Time</div>
                </div>
                <div className="did-summary-divider" style={{ width: "100%", height: "1px" }} />
                <div className="did-summary-stat">
                  <div className="did-summary-stat-value"><Counter target={totalClauses} /></div>
                  <div className="did-summary-stat-label">Total Clauses</div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* ── Section 2: Risk Analysis ── */}
          <motion.div
            className="did-card"
            custom={5}
            initial="hidden"
            animate="visible"
            variants={cardVariants}
          >
            <div className="did-card-header">
              <div className="did-card-title-row">
                <div className="did-card-icon"><Shield size={14} /></div>
                <span className="did-card-title">Risk Analysis</span>
              </div>
              <motion.button
                className="did-btn did-btn--ghost"
                onClick={() => navigate("/upload")}
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.97 }}
              >
                Full Report <ChevronRight size={12} />
              </motion.button>
            </div>

            <div className="did-risk-layout">
              <RiskGauge risk={riskLevel} />
              <div className="did-risk-list">
                {RISKS.map((r, i) => (
                  <motion.div
                    key={i}
                    className="did-risk-item"
                    initial={{ opacity: 0, x: 16 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 + i * 0.1, duration: 0.45 }}
                  >
                    <div className={`did-risk-dot did-risk-dot--${r.severity}`} />
                    <div className="did-risk-item-text">
                      <div className="did-risk-item-title">{r.title}</div>
                      <div className="did-risk-item-desc">{r.desc}</div>
                    </div>
                    <span className={`did-risk-severity did-risk-severity--${r.severity}`}>
                      {r.severity}
                    </span>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* ── Section 3: Clause Analysis ── */}
          <motion.div
            className="did-card"
            custom={6}
            initial="hidden"
            animate="visible"
            variants={cardVariants}
          >
            <div className="did-card-header">
              <div className="did-card-title-row">
                <div className="did-card-icon"><List size={14} /></div>
                <span className="did-card-title">Clause Analysis</span>
              </div>
              <span className="did-clause-count">
                <TrendingUp size={10} />
                {totalClauses} clauses
              </span>
            </div>

            <div className="did-clause-chips">
              {CLAUSES.map((c) => (
                <ClauseChip
                  key={c.label}
                  label={c.label}
                  count={c.count}
                  active={activeClause === c.label}
                  onClick={() => setActiveClause(activeClause === c.label ? null : c.label)}
                />
              ))}
            </div>

            <motion.button
              className="did-btn did-btn--ghost"
              onClick={() => navigate("/upload")}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              style={{ marginTop: "auto" }}
            >
              View Detailed Analysis <ChevronRight size={12} />
            </motion.button>
          </motion.div>

          {/* ── Section 4: AI Assistant ── */}
          <motion.div
            className="did-card"
            custom={7}
            initial="hidden"
            animate="visible"
            variants={cardVariants}
          >
            <div className="did-card-header">
              <div className="did-card-title-row">
                <div className="did-card-icon"><MessageSquare size={14} /></div>
                <span className="did-card-title">AI Assistant</span>
              </div>
              <div className="did-badge did-badge--status" style={{ fontSize: "9px" }}>
                Ready
              </div>
            </div>

            {/* Chat preview */}
            <div className="did-chat-preview">
              <div className="did-chat-msg">
                <div className="did-chat-avatar did-chat-avatar--user">U</div>
                <div className="did-chat-bubble did-chat-bubble--user">
                  What are the key risks in this contract?
                </div>
              </div>
              <div className="did-chat-msg">
                <div className="did-chat-avatar did-chat-avatar--ai">AI</div>
                <div className="did-chat-bubble did-chat-bubble--ai">
                  I've identified 3 primary risks: an unlimited liability clause, ambiguous termination terms, and missing jurisdiction definition. The liability exposure is the most critical concern…
                </div>
              </div>
            </div>

            {/* Suggested prompts */}
            <div className="did-prompt-pills">
              {PROMPTS.map((p) => (
                <motion.button
                  key={p}
                  className="did-prompt-pill"
                  onClick={() => navigate("/upload")}
                  whileHover={{ scale: 1.04 }}
                  whileTap={{ scale: 0.97 }}
                >
                  <Sparkles size={10} />
                  {p}
                </motion.button>
              ))}
            </div>

            <motion.button
              className="did-btn did-btn--gold"
              onClick={() => navigate("/upload")}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              style={{ width: "100%", justifyContent: "center" }}
            >
              <MessageSquare size={13} />
              Open AI Assistant
            </motion.button>
          </motion.div>

          {/* ── Section 5: Recommendations (full width) ── */}
          <motion.div
            className="did-card did-main-grid--wide"
            custom={8}
            initial="hidden"
            animate="visible"
            variants={cardVariants}
          >
            <div className="did-card-header">
              <div className="did-card-title-row">
                <div className="did-card-icon"><Lightbulb size={14} /></div>
                <span className="did-card-title">AI Recommendations</span>
              </div>
              <span style={{ fontSize: "11px", color: "var(--muted)", letterSpacing: "0.05em" }}>
                {RECOMMENDATIONS.length} suggestions
              </span>
            </div>

            <div className="did-rec-list">
              {RECOMMENDATIONS.map((r, i) => (
                <motion.div
                  key={i}
                  className="did-rec-item"
                  initial={{ opacity: 0, x: -16 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.7 + i * 0.08, duration: 0.4 }}
                >
                  <div className="did-rec-bullet" />
                  <div className="did-rec-text">{r.text}</div>
                  <span className={`did-rec-tag did-rec-tag--${r.tag}`}>
                    {r.tag}
                  </span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* ── Bottom Quick Actions ── */}
        <motion.div
          custom={9}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="did-quick-actions">
            {[
              { icon: FileSearch,    label: "View Summary",     action: () => navigate("/upload") },
              { icon: List,         label: "Clause Analysis",  action: () => navigate("/upload") },
              { icon: MessageSquare, label: "AI Assistant",    action: () => navigate("/upload") },
              { icon: Download,     label: "PDF Report",       action: () => alert("PDF export coming soon!") },
            ].map(({ icon: Icon, label, action }, i) => (
              <motion.button
                key={label}
                className="did-qa-btn"
                onClick={action}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 + i * 0.07, duration: 0.45 }}
                whileHover={{ y: -4, scale: 1.02 }}
                whileTap={{ scale: 0.97 }}
              >
                <div className="did-qa-icon"><Icon size={18} /></div>
                <span className="did-qa-label">{label}</span>
              </motion.button>
            ))}
          </div>
        </motion.div>

      </div>
    </div>
  );
}
