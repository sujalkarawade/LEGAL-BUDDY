import { useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { motion, useMotionValue, useSpring, type Transition } from "framer-motion";
import {
  ArrowLeft,
  FileText,
  Gavel,
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





      </div>
    </div>
  );
}
