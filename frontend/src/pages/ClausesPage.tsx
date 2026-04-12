import { motion, AnimatePresence } from "framer-motion";
import { List, Loader2, Target, AlertTriangle, UserCheck, ScanSearch } from "lucide-react";
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

export default function ClausesPage({ analysis, handleAnalyze, loading, embedded, error }: ClausesPageProps) {
  if (!embedded) {
    return (
      <div className="no-document-state">
        <div className="no-document-icon"><List size={32} /></div>
        <h2 className="no-document-title">No Document Embedded</h2>
        <p className="no-document-description">Upload and embed a document first to analyze clauses.</p>
      </div>
    );
  }

  const getRiskColor = (risk: string | undefined): string => {
    switch (risk?.toLowerCase()) {
      case "high":    return "risk-high";
      case "medium":  return "risk-medium";
      case "low":     return "risk-low";
      default:        return "risk-unknown";
    }
  };

  const hasAnalysis = !!analysis;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className={`clauses-page ${hasAnalysis ? "clauses-page--has-content" : "clauses-page--empty"}`}
    >
      <AnimatePresence mode="wait">
        {!hasAnalysis && !loading ? (
          /* ── Hero empty state ── */
          <motion.div
            key="hero"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.35 }}
            className="clauses-hero"
          >
            <div className="hero-dots">
              {Array.from({ length: 16 }).map((_, i) => (
                <div key={i} className="hero-dot" />
              ))}
            </div>

            <div className="hero-icon-ring">
              <div className="hero-icon-inner">
                <ScanSearch size={26} />
              </div>
            </div>

            <div className="hero-badge">
              <span className="hero-badge-dot" />
              Risk Detection
            </div>

            <h1 className="clauses-title">Clause Analysis</h1>
            <p className="clauses-hero-description">
              Detect clauses, assess risk levels, identify missing provisions, and get lawyer recommendations.
            </p>

            <button onClick={handleAnalyze} disabled={loading} className="analyze-button disabled:opacity-50">
              <Target size={16} />
              Run Analysis
            </button>
          </motion.div>
        ) : loading ? (
          /* ── Loading state ── */
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="clauses-hero"
          >
            <div className="hero-icon-ring">
              <div className="hero-icon-inner">
                <Loader2 size={26} className="animate-spin" />
              </div>
            </div>
            <h1 className="clauses-title">Running Analysis</h1>
            <p className="clauses-hero-description">Scanning clauses, assessing risks, and building insights…</p>
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
            <div className="clauses-header">
              <div>
                <h1 className="clauses-title">Clause Analysis</h1>
                <p className="clauses-description">Deep insights into document clauses, risk factors, and legal advice.</p>
              </div>
              <button onClick={handleAnalyze} disabled={loading} className="analyze-button disabled:opacity-50">
                <Target size={14} />
                Re-analyze
              </button>
            </div>

            {error && <div className="error-message">{error}</div>}

            <div className="analysis-content">
              {/* Detected Clauses */}
              <section className="detected-clauses">
                <h2 className="section-title">
                  <List size={18} className="section-icon" /> Detected Clauses
                </h2>
                <div className="clause-tags">
                  {analysis!.detected_clauses.map((c) => (
                    <div key={c} className={`clause-tag ${getRiskColor(analysis!.clause_risks[c])}`}>
                      {c} <span className="risk-level">({analysis!.clause_risks[c] || "unknown"})</span>
                    </div>
                  ))}
                </div>
              </section>

              <div className="analysis-grid">
                <section className="advice-section">
                  <div>
                    <h2 className="section-title">
                      <AlertTriangle size={18} className="section-icon-warning" /> Missing Clauses
                    </h2>
                    <div className="advice-content">
                      {analysis!.advice_missing || "No missing clauses detected."}
                    </div>
                  </div>
                  {analysis!.advice_unusual && (
                    <div>
                      <h2 className="section-title">
                        <AlertTriangle size={18} className="section-icon-danger" /> Unusual Phrases
                      </h2>
                      <div className="unusual-content">{analysis!.advice_unusual}</div>
                    </div>
                  )}
                </section>

                {analysis!.top_lawyers && analysis!.top_lawyers.length > 0 && (
                  <section className="lawyers-section">
                    <h2 className="section-title">
                      <UserCheck size={18} className="section-icon-purple" /> Recommended Lawyers
                    </h2>
                    <div className="lawyers-list">
                      {analysis!.top_lawyers.map((l, i) => (
                        <div key={i} className="lawyer-card">
                          <div>
                            <p className="lawyer-name">{l.name}</p>
                            <p className="lawyer-specialization">{l.specialization}</p>
                          </div>
                          <div className="lawyer-experience">
                            <p className="experience-years">{l.experience} yrs</p>
                            <p className="experience-label">experience</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                )}
              </div>

              {analysis!.co_occurrence && analysis!.co_occurrence.length > 0 && (
                <section className="co-occurrence-section">
                  <h2 className="section-title">Co-occurrence Analysis</h2>
                  <div className="table-container">
                    <table className="co-occurrence-table">
                      <thead>
                        <tr>
                          <th className="table-header">Clause A</th>
                          <th className="table-header">Clause B</th>
                          <th className="table-header table-header-center">Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {analysis!.co_occurrence.sort((a, b) => b.count - a.count).map((row, i) => (
                          <tr key={i} className="table-row">
                            <td className="table-cell">{row.a}</td>
                            <td className="table-cell">{row.b}</td>
                            <td className="table-cell table-cell-center">
                              <span className="count-badge">{row.count}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
