import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { motion, useMotionValue, useSpring, useTransform, AnimatePresence } from "framer-motion";
import { Scale, Upload, FileText, Shield, Search, Wand2, MessageSquare, CheckCircle, ArrowRight, Sparkles, Zap, Clock } from "lucide-react";
import "./Dashboard.css";

/* ─── Floating Particle ─────────────────────────────────────── */
function Particle({ x, y, delay }: { x: number; y: number; delay: number }) {
  return (
    <motion.div
      className="particle"
      style={{ left: `${x}%`, top: `${y}%` }}
      animate={{
        y: [0, -30, 0],
        opacity: [0, 1, 0],
        scale: [0.5, 1.2, 0.5],
      }}
      transition={{
        duration: 3 + Math.random() * 2,
        repeat: Infinity,
        delay,
        ease: "easeInOut",
      }}
    />
  );
}

/* ─── Legal Symbol Floater ──────────────────────────────────── */
function LegalSymbol({ symbol, x, y, delay, size }: { symbol: string; x: number; y: number; delay: number; size: number }) {
  return (
    <motion.div
      className="legal-symbol"
      style={{ left: `${x}%`, top: `${y}%`, fontSize: size }}
      animate={{
        y: [0, -20, 0],
        opacity: [0.03, 0.08, 0.03],
        rotate: [0, 5, -5, 0],
      }}
      transition={{
        duration: 8 + delay * 2,
        repeat: Infinity,
        delay,
        ease: "easeInOut",
      }}
    >
      {symbol}
    </motion.div>
  );
}

/* ─── Golden Light Streak ───────────────────────────────────── */
function LightStreak({ delay, top }: { delay: number; top: number }) {
  return (
    <motion.div
      className="light-streak"
      style={{ top: `${top}%` }}
      animate={{
        x: ["-100%", "200%"],
        opacity: [0, 0.6, 0],
      }}
      transition={{
        duration: 4,
        repeat: Infinity,
        delay,
        ease: "easeInOut",
      }}
    />
  );
}

/* ─── Badge Component ───────────────────────────────────────── */
const badges = [
  { icon: FileText, label: "Contract Analysis" },
  { icon: Search, label: "Clause Detection" },
  { icon: Shield, label: "Risk Assessment" },
  { icon: Wand2, label: "Legal Drafting" },
  { icon: MessageSquare, label: "AI Assistant" },
];

/* ─── Floating Stat Card ────────────────────────────────────── */
function FloatingCard({ value, label, icon: Icon, delay, className }: {
  value: string; label: string; icon: React.ElementType; delay: number; className: string;
}) {
  return (
    <motion.div
      className={`floating-card ${className}`}
      initial={{ opacity: 0, scale: 0.8, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ delay, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
    >
      <motion.div
        animate={{ y: [0, -8, 0] }}
        transition={{ duration: 4 + delay, repeat: Infinity, ease: "easeInOut" }}
      >
        <div className="floating-card-inner">
          <div className="floating-card-icon">
            <Icon size={16} />
          </div>
          <div>
            <div className="floating-card-value">{value}</div>
            <div className="floating-card-label">{label}</div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

/* ─── Animated Counter ──────────────────────────────────────── */
function AnimatedCounter({ target, suffix = "" }: { target: number; suffix?: string }) {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    let hasStarted = false;

    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !hasStarted) {
        hasStarted = true;

        // Wait for the container's fade-in delay (2.2s) before counting
        setTimeout(() => {
          let start = 0;
          const duration = 2000;
          const step = (target / duration) * 16;

          timer = setInterval(() => {
            start += step;
            if (start >= target) {
              setCount(target);
              clearInterval(timer);
            } else {
              setCount(Math.floor(start));
            }
          }, 16);
        }, 2200);
      }
    });
    if (ref.current) observer.observe(ref.current);
    return () => {
      observer.disconnect();
      if (timer) clearInterval(timer);
    };
  }, [target]);

  return <span ref={ref}>{count.toLocaleString()}{suffix}</span>;
}

/* ─── Command Panel ──────────────────────────────────────────── */
function CommandPanel({
  icon: Icon,
  title,
  subtitle,
  onNavigate,
  variant,
  delay,
}: {
  icon: React.ElementType;
  title: string;
  subtitle: string;
  onNavigate: () => void;
  variant: "upload" | "generate";
  delay: number;
}) {
  const [isHovered, setIsHovered] = useState(false);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const panelRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = panelRef.current?.getBoundingClientRect();
    if (!rect) return;
    setMousePos({
      x: ((e.clientX - rect.left) / rect.width) * 100,
      y: ((e.clientY - rect.top) / rect.height) * 100,
    });
  };

  const particles = Array.from({ length: 8 }, (_, i) => ({
    x: 10 + i * 12,
    delay: i * 0.3,
  }));

  const isUpload = variant === "upload";

  return (
    <motion.div
      ref={panelRef}
      className={`cmd-panel cmd-panel--${variant}`}
      initial={{ opacity: 0, x: 80, y: 20 }}
      animate={{ opacity: 1, x: 0, y: 0 }}
      transition={{ delay, duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
      onClick={onNavigate}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onMouseMove={handleMouseMove}
      whileHover={{ y: -4, scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      role="button"
      tabIndex={0}
      aria-label={title}
      onKeyDown={e => e.key === "Enter" && onNavigate()}
    >
      {/* Dynamic cursor glow */}
      <motion.div
        className="cmd-cursor-glow"
        animate={{
          opacity: isHovered ? 1 : 0,
          left: `${mousePos.x}%`,
          top: `${mousePos.y}%`,
        }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      />



      {/* Light sweep */}
      <motion.div
        className="cmd-light-sweep"
        animate={isHovered ? {
          x: ["-100%", "200%"],
          opacity: [0, 0.15, 0],
        } : { opacity: 0 }}
        transition={{ duration: 1.5, repeat: isHovered ? Infinity : 0, repeatDelay: 2, ease: "easeInOut" }}
      />

      {/* Content */}
      <div className="cmd-content">
        {/* Icon */}
        <motion.div
          className="cmd-icon-wrap"
          animate={{
            rotate: isHovered ? (isUpload ? [0, -3, 3, 0] : [0, 3, -3, 0]) : 0,
            scale: isHovered ? 1.05 : 1,
          }}
          transition={{ duration: 0.6, ease: "easeInOut" }}
        >
          <div className="cmd-icon-ring cmd-icon-ring--outer" />
          <div className="cmd-icon-ring cmd-icon-ring--inner" />
          <Icon size={36} strokeWidth={1.2} className="cmd-icon" />
        </motion.div>

        {/* Title */}
        <div className="cmd-title">{title}</div>

        {/* Subtitle */}
        <motion.div
          className="cmd-subtitle"
          animate={{ opacity: isHovered ? 1 : 0.55, y: isHovered ? 0 : 4 }}
          transition={{ duration: 0.3 }}
        >
          {subtitle}
        </motion.div>

        {/* Hover arrow indicator */}
        <motion.div
          className="cmd-arrow"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: isHovered ? 1 : 0, y: isHovered ? 0 : 8 }}
          transition={{ duration: 0.25 }}
        >
          <ArrowRight size={16} />
          <span>OPEN</span>
        </motion.div>
      </div>

      {/* Bottom energy bar */}
      <motion.div
        className="cmd-energy-bar"
        animate={{ scaleX: isHovered ? 1 : 0.3, opacity: isHovered ? 1 : 0.3 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
      />
    </motion.div>
  );
}

/* ─── Right Panel Wrapper ────────────────────────────────────── */
function CommandPanels({ onUpload, onGenerate }: { onUpload: () => void; onGenerate: () => void }) {
  return (
    <div className="cmd-panels">
      <CommandPanel
        icon={Upload}
        title="UPLOAD"
        subtitle="Analyze Contracts & Legal Documents"
        onNavigate={onUpload}
        variant="upload"
        delay={0.4}
      />
      <CommandPanel
        icon={Wand2}
        title="GENERATE"
        subtitle="Create Legal Drafts with AI"
        onNavigate={onGenerate}
        variant="generate"
        delay={0.4}
      />
    </div>
  );
}


/* ─── MAIN DASHBOARD ────────────────────────────────────────── */
export default function Dashboard() {
  const navigate = useNavigate();
  const containerRef = useRef<HTMLDivElement>(null);

  // Cursor spotlight
  const cursorX = useMotionValue(0);
  const cursorY = useMotionValue(0);
  const springX = useSpring(cursorX, { stiffness: 80, damping: 20 });
  const springY = useSpring(cursorY, { stiffness: 80, damping: 20 });

  // Parallax values
  const parallaxX = useTransform(springX, [0, window.innerWidth], [-15, 15]);
  const parallaxY = useTransform(springY, [0, window.innerHeight], [-15, 15]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    cursorX.set(e.clientX);
    cursorY.set(e.clientY);
  }, [cursorX, cursorY]);

  useEffect(() => {
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [handleMouseMove]);

  // Badge animation
  const [activeBadge, setActiveBadge] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setActiveBadge(p => (p + 1) % badges.length), 1800);
    return () => clearInterval(t);
  }, []);

  const heroLetters = ["L", "E", "G", "A", "L"];
  const buddyLetters = ["B", "U", "D", "D", "Y"];

  return (
    <div className="dashboard-v2" ref={containerRef}>
      {/* ── Cursor Spotlight ── */}
      <motion.div
        className="cursor-spotlight"
        style={{ x: springX, y: springY }}
      />

      {/* ── Animated Background ── */}
      <div className="bg-layer">
        {/* Grid */}
        <div className="bg-grid" />
        {/* Noise texture */}
        <div className="bg-noise" />
        {/* Radial glow */}
        <div className="bg-radial-gold" />
        <div className="bg-radial-blue" />

        {/* Legal symbols */}
        <LegalSymbol symbol="⚖️" x={8} y={15} delay={0} size={80} />
        <LegalSymbol symbol="§" x={85} y={70} delay={1} size={120} />
        <LegalSymbol symbol="⚖️" x={70} y={10} delay={2} size={60} />
        <LegalSymbol symbol="§" x={20} y={75} delay={0.5} size={90} />
        <LegalSymbol symbol="©" x={50} y={85} delay={1.5} size={70} />
        <LegalSymbol symbol="™" x={90} y={30} delay={3} size={60} />

        {/* Light streaks */}
        <LightStreak delay={0} top={20} />
        <LightStreak delay={3} top={55} />
        <LightStreak delay={6} top={80} />

        {/* Parallax layer */}
        <motion.div className="parallax-layer" style={{ x: parallaxX, y: parallaxY }}>
          <div className="parallax-orb parallax-orb-1" />
          <div className="parallax-orb parallax-orb-2" />
        </motion.div>
      </div>

      {/* ── Main Content ── */}
      <div className="dashboard-v2-inner">

        {/* ── Split Content ── */}
        <div className="lb-split">

          {/* LEFT: Hero */}
          <div className="lb-left">

            {/* Tag */}
            <motion.div
              className="lb-tag"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.6 }}
            >
              <span>Powered by Advanced AI</span>
            </motion.div>

            {/* Hero text: LEGAL */}
            <div className="lb-hero-title">
              <div className="lb-word lb-word-legal">
                {heroLetters.map((l, i) => (
                  <motion.span
                    key={i}
                    className="lb-letter"
                    initial={{ opacity: 0, x: -80 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{
                      delay: 0.4 + i * 0.06,
                      duration: 0.7,
                      ease: [0.22, 1, 0.36, 1],
                    }}
                  >
                    {l}
                  </motion.span>
                ))}
              </div>

              {/* BUDDY slides from bottom */}
              <div className="lb-word lb-word-buddy">
                {buddyLetters.map((l, i) => (
                  <motion.span
                    key={i}
                    className="lb-letter lb-letter-gold"
                    initial={{ opacity: 0, y: 80 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{
                      delay: 0.7 + i * 0.07,
                      duration: 0.7,
                      ease: [0.22, 1, 0.36, 1],
                    }}
                  >
                    {l}
                  </motion.span>
                ))}
              </div>

              {/* Gold underline grows */}
              <motion.div
                className="lb-underline"
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ delay: 1.2, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
              />
            </div>

            {/* Tagline */}
            <motion.p
              className="lb-tagline"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.4, duration: 0.7, ease: "easeOut" }}
            >
              AI-powered legal intelligence platform
            </motion.p>

            {/* Badges */}
            <motion.div
              className="lb-badges"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.6, duration: 0.5 }}
            >
              {badges.map(({ icon: Icon, label }, i) => (
                <motion.div
                  key={i}
                  className={`lb-badge ${activeBadge === i ? "lb-badge-active" : ""}`}
                  animate={{
                    scale: activeBadge === i ? 1.06 : 1,
                    opacity: activeBadge === i ? 1 : 0.55,
                  }}
                  transition={{ duration: 0.3 }}
                  whileHover={{ scale: 1.1, opacity: 1 }}
                >
                  <Icon size={11} />
                  <span>{label}</span>
                </motion.div>
              ))}
            </motion.div>


          </div>

          {/* RIGHT: Command Panels */}
          <div className="lb-right">
            <CommandPanels
              onUpload={() => navigate("/upload")}
              onGenerate={() => navigate("/generate")}
            />
          </div>
        </div>

        {/* ── Stats Bar ── */}
        <motion.div
          className="lb-stats"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 2.2, duration: 0.7, ease: "easeOut" }}
        >
          {[
            { value: 95, suffix: "%", label: "Risk Detection Accuracy", icon: Shield },
            { value: 50, suffix: "+", label: "Clause Types", icon: FileText },
            { value: 24, suffix: "/7", label: "AI Legal Assistance", icon: Clock },
          ].map(({ value, suffix, label, icon: Icon }, i) => (
            <div key={i} className="lb-stat">
              <div className="lb-stat-icon"><Icon size={16} /></div>
              <div className="lb-stat-value">
                <AnimatedCounter target={value} suffix={suffix} />
              </div>
              <div className="lb-stat-label">{label}</div>
            </div>
          ))}
        </motion.div>
      </div>
    </div>
  );
}
