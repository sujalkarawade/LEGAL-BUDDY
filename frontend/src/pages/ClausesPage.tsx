import { motion, AnimatePresence } from "framer-motion";
import { useState, useMemo } from "react";
import { List, Loader2, Target, AlertTriangle, UserCheck, ScanSearch,
         ShieldAlert, ShieldCheck, ShieldQuestion, Zap, Search, X, Copy, CheckCircle2 } from "lucide-react";
import { mitigateClause, MitigateResponse } from "../api";
import "./ClausesPage.css";

interface ClauseRisks { [key: string]: string; }
interface CoOccurrence { a: string; b: string; count: number; }
interface Lawyer { name: string; specialization: string; experience: number; }
interface Analysis {
  detected_clauses: string[];
  clause_risks: ClauseRisks;
  advice_missing: string;
  advice_unusual?: string;
  top_lawyers?: Lawyer[];
  co_occurrence?: CoOccurrence[];
}

interface ClausesPageProps {
  analysis: Analysis | null;
  handleAnalyze: () => void;
  loading: boolean;
  embedded: boolean;
  error: string;
}

const RISK_META: Record<string, { label: string; icon: React.ReactNode; weight: number }> = {
  high:    { label: "High",    icon: <ShieldAlert size={13} />,   weight: 0 },
  medium:  { label: "Medium",  icon: <AlertTriangle size={13} />, weight: 50 },
  low:     { label: "Low",     icon: <ShieldCheck size={13} />,   weight: 100 },
  unknown: { label: "Unknown", icon: <ShieldQuestion size={13} />, weight: 50 },
};

const FILTER_OPTIONS = ["all", "high", "medium", "low", "unknown"] as const;
type FilterOption = typeof FILTER_OPTIONS[number];

function getRiskKey(risk: string | undefined) {
  const k = risk?.toLowerCase() ?? "unknown";
  return RISK_META[k] ? k : "unknown";
}

/** 0–100 score: 100 = all low, 0 = all high */
function calcRiskScore(clauses: string[], risks: ClauseRisks): number {
  if (!clauses.length) return 0;
  const sum = clauses.reduce((acc, c) => acc + RISK_META[getRiskKey(risks[c])].weight, 0);
  return Math.round(sum / clauses.length);
}

/** SVG circle gauge */
function RiskGauge({ score }: { score: number }) {
  const r = 44;
  const circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  const color = score >= 70 ? "#22c55e" : score >= 40 ? "#f59e0b" : "#ef4444";
  const label = score >= 70 ? "Low Risk" : score >= 40 ? "Moderate" : "High Risk";

  return (
    <div className="gauge-wrap">
      <svg width="110" height="110" viewBox="0 0 110 110">
        {/* track */}
        <circle cx="55" cy="55" r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
        {/* progress */}
        <circle
          cx="55" cy="55" r={r} fill="none"
          stroke={color} strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circ}`}
          strokeDashoffset={circ * 0.25}   /* start from top */
          style={{ transition: "stroke-dasharray 1s ease" }}
        />
      </svg>
      <div className="gauge-center">
        <span className="gauge-score" style={{ color }}>{score}</span>
        <span className="gauge-label" style={{ color }}>{label}</span>
      </div>
    </div>
  );
}

export default function ClausesPage({ analysis, handleAnalyze, loading, embedded, error }: ClausesPageProps) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<FilterOption>("all");

  const [mitigating, setMitigating] = useState<string | null>(null);
  const [mitigationData, setMitigationData] = useState<MitigateResponse | null>(null);
  const [mitigateError, setMitigateError] = useState<string>("");
  const [copied, setCopied] = useState(false);

  const handleMitigate = async (clause: string) => {
    setMitigating(clause);
    setMitigationData(null);
    setMitigateError("");
    try {
      const data = await mitigateClause(clause);
      setMitigationData(data);
    } catch (e: any) {
      setMitigateError(e.message);
    } finally {
      setMitigating(null);
    }
  };

  const handleCopy = () => {
    if (mitigationData?.mitigated_text) {
      navigator.clipboard.writeText(mitigationData.mitigated_text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!embedded) {
    return (
      <div className="no-document-state">
        <div className="no-document-icon"><List size={32} /></div>
        <h2 className="no-document-title">No Document Embedded</h2>
        <p className="no-document-description">Upload and embed a document first to analyze clauses.</p>
      </div>
    );
  }

  const hasAnalysis = !!analysis;

  const riskCounts = useMemo(() => hasAnalysis
    ? analysis!.detected_clauses.reduce((acc, c) => {
        const k = getRiskKey(analysis!.clause_risks[c]);
        acc[k] = (acc[k] || 0) + 1;
        return acc;
      }, {} as Record<string, number>)
    : {}, [analysis]);

  const total = hasAnalysis ? analysis!.detected_clauses.length : 0;
  const riskScore = hasAnalysis ? calcRiskScore(analysis!.detected_clauses, analysis!.clause_risks) : 0;

  const filteredClauses = useMemo(() => {
    if (!hasAnalysis) return [];
    return analysis!.detected_clauses.filter(c => {
      const matchesFilter = filter === "all" || getRiskKey(analysis!.clause_risks[c]) === filter;
      const matchesSearch = c.replace(/_/g, " ").toLowerCase().includes(search.toLowerCase());
      return matchesFilter && matchesSearch;
    });
  }, [analysis, filter, search]);

  const missingItems = hasAnalysis && analysis!.advice_missing
    ? analysis!.advice_missing.split(/[,;]/).map(s => s.trim()).filter(Boolean)
    : [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className={`clauses-page ${hasAnalysis ? "clauses-page--has-content" : "clauses-page--empty"}`}
    >
      <AnimatePresence mode="wait">
        {!hasAnalysis && !loading ? (
          <motion.div
            key="hero"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.35 }}
            className="clauses-hero"
          >
            <div className="hero-dots">
              {Array.from({ length: 16 }).map((_, i) => <div key={i} className="hero-dot" />)}
            </div>
            <div className="hero-icon-ring">
              <div className="hero-icon-inner"><ScanSearch size={26} /></div>
            </div>
            <div className="hero-badge"><span className="hero-badge-dot" />Risk Detection</div>
            <h1 className="clauses-title">Clause Analysis</h1>
            <p className="clauses-hero-description">
              Detect clauses, assess risk levels, identify missing provisions, and get lawyer recommendations.
            </p>
            <button onClick={handleAnalyze} disabled={loading} className="analyze-button disabled:opacity-50">
              <Target size={16} /> Run Analysis
            </button>
          </motion.div>

        ) : loading ? (
          <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="clauses-hero">
            <div className="hero-icon-ring">
              <div className="hero-icon-inner"><Loader2 size={26} className="animate-spin" /></div>
            </div>
            <h1 className="clauses-title">Running Analysis</h1>
            <p className="clauses-hero-description">Scanning clauses, assessing risks, and building insights…</p>
          </motion.div>

        ) : (
          <motion.div
            key="content"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
            style={{ width: "100%" }}
          >
            {/* Header */}
            <div className="clauses-header">
              <div>
                <h1 className="clauses-title">Clause Analysis</h1>
                <p className="clauses-description">Deep insights into document clauses, risk factors, and legal advice.</p>
              </div>
              <button onClick={handleAnalyze} disabled={loading} className="analyze-button disabled:opacity-50">
                <Target size={14} /> Re-analyze
              </button>
            </div>

            {error && <div className="error-message">{error}</div>}

            {/* ── Top row: Risk Summary Bar + Gauge ── */}
            <div className="top-row">
              <div className="risk-summary-bar">
                <div className="risk-summary-total">
                  <span className="risk-total-number">{total}</span>
                  <span className="risk-total-label">Clauses</span>
                </div>
                <div className="risk-summary-divider" />
                {(["high","medium","low","unknown"] as const).map(k => (
                  <div
                    key={k}
                    className={`risk-summary-item risk-summary-${k} ${filter === k ? "risk-summary-item--active" : ""}`}
                    onClick={() => setFilter(f => f === k ? "all" : k)}
                    title={`Filter by ${k} risk`}
                  >
                    {RISK_META[k].icon}
                    <span className="risk-summary-count">{riskCounts[k] || 0}</span>
                    <span className="risk-summary-label">{RISK_META[k].label}</span>
                  </div>
                ))}
              </div>

              {/* Risk Score Gauge */}
              <div className="gauge-card">
                <RiskGauge score={riskScore} />
                <p className="gauge-title">Risk Score</p>
              </div>
            </div>

            <div className="analysis-content">

              {/* ── Clause Cards with search + filter ── */}
              <section className="detected-clauses">
                <div className="clauses-toolbar">
                  <h2 className="section-title" style={{ margin: 0 }}>
                    <List size={18} className="section-icon" /> Detected Clauses
                  </h2>
                  <div className="clause-search-wrap">
                    <Search size={13} className="clause-search-icon" />
                    <input
                      type="text"
                      className="clause-search"
                      placeholder="Search clauses…"
                      value={search}
                      onChange={e => setSearch(e.target.value)}
                    />
                    {search && (
                      <button className="clause-search-clear" onClick={() => setSearch("")}>
                        <X size={12} />
                      </button>
                    )}
                  </div>
                  {filter !== "all" && (
                    <button className="filter-clear-btn" onClick={() => setFilter("all")}>
                      <X size={11} /> Clear filter
                    </button>
                  )}
                </div>

                {filteredClauses.length === 0 ? (
                  <p className="advice-empty" style={{ marginTop: "0.75rem" }}>No clauses match your search.</p>
                ) : (
                  <div className="clause-cards">
                    {filteredClauses.map((c, i) => {
                      const rk = getRiskKey(analysis!.clause_risks[c]);
                      const meta = RISK_META[rk];
                      return (
                        <motion.div
                          key={c}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.03 }}
                          className={`clause-card clause-card--${rk} clause-card--clickable`}
                          onClick={() => handleMitigate(c)}
                        >
                          <div className="clause-card-icon">{meta.icon}</div>
                          <div className="clause-card-body">
                            <span className="clause-card-name">{c.replace(/_/g, " ")}</span>
                            <span className={`clause-card-risk clause-card-risk--${rk}`}>{meta.label} Risk</span>
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                )}
              </section>

              <div className="analysis-grid">
                {/* ── Missing Clauses ── */}
                <section className="advice-section">
                  <h2 className="section-title">
                    <AlertTriangle size={18} className="section-icon-warning" /> Missing Clauses
                  </h2>
                  <div className="missing-pills">
                    {missingItems.length > 0
                      ? missingItems.map((item, i) => (
                          <div key={i} className="missing-pill">
                            <Zap size={11} className="missing-pill-icon" />
                            {item}
                          </div>
                        ))
                      : <p className="advice-empty">No missing clauses detected.</p>
                    }
                  </div>
                </section>

                {/* ── Unusual Phrases ── */}
                <section className="advice-section">
                  <h2 className="section-title">
                    <AlertTriangle size={18} className="section-icon-danger" /> Unusual Phrases
                  </h2>
                  {analysis!.advice_unusual
                    ? <div className="unusual-content">{analysis!.advice_unusual}</div>
                    : <p className="advice-empty">No unusual phrases detected.</p>
                  }
                </section>
              </div>

              {/* ── Lawyers ── */}
              {analysis!.top_lawyers && analysis!.top_lawyers.length > 0 && (
                <section className="lawyers-section">
                  <h2 className="section-title">
                    <UserCheck size={18} className="section-icon-purple" /> Recommended Lawyers
                  </h2>
                  <div className="lawyers-list">
                    {analysis!.top_lawyers.map((l, i) => (
                      <div key={i} className="lawyer-card">
                        <div className="lawyer-rank">#{i + 1}</div>
                        <div className="lawyer-info">
                          <p className="lawyer-name">{l.name}</p>
                          <p className="lawyer-specialization">{l.specialization}</p>
                        </div>
                        <div className="lawyer-experience">
                          <p className="experience-years">{l.experience}</p>
                          <p className="experience-label">yrs exp</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}

            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {(mitigating || mitigationData || mitigateError) && (
          <motion.div
            className="mitigation-modal-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => {
              if (!mitigating) {
                setMitigationData(null);
                setMitigateError("");
              }
            }}
          >
            <motion.div
              className="mitigation-modal"
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.95 }}
              onClick={e => e.stopPropagation()}
            >
              <div className="mitigation-modal-header">
                <h2>Clause Mitigation</h2>
                <button className="mitigation-close" onClick={() => {
                  setMitigationData(null);
                  setMitigateError("");
                  setMitigating(null);
                }}>
                  <X size={20} />
                </button>
              </div>
              
              <div className="mitigation-modal-body">
                {mitigating ? (
                  <div className="mitigation-loading">
                    <Loader2 size={32} className="animate-spin" />
                    <p>Analyzing {mitigating.replace(/_/g, " ")} and drafting mitigation...</p>
                  </div>
                ) : mitigateError ? (
                  <div className="mitigation-error">
                    <AlertTriangle size={32} />
                    <p>{mitigateError}</p>
                  </div>
                ) : mitigationData ? (
                  <div className="mitigation-columns">
                    <div className="mitigation-col mitigation-col-original">
                      <h3>Original Clause & Risk</h3>
                      <div className="mitigation-box original-box">
                        <p className="mitigation-text">{mitigationData.original_text}</p>
                      </div>
                      <div className="mitigation-risk">
                        <h4><AlertTriangle size={14} /> Risk Analysis</h4>
                        <p>{mitigationData.explanation}</p>
                      </div>
                    </div>
                    
                    <div className="mitigation-col mitigation-col-mitigated">
                      <div className="mitigated-header">
                        <h3>Safe Alternative</h3>
                        <button className="copy-btn" onClick={handleCopy}>
                          {copied ? <CheckCircle2 size={14} /> : <Copy size={14} />}
                          {copied ? "Copied!" : "Copy"}
                        </button>
                      </div>
                      <div className="mitigation-box mitigated-box">
                        <p className="mitigation-text">{mitigationData.mitigated_text}</p>
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
