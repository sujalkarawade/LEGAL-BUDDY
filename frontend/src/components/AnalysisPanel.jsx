export default function AnalysisPanel({ data }) {
  const { detected_clauses, clause_risks, co_occurrence, advice_missing, advice_unusual, top_lawyers } = data;

  return (
    <section className="card">
      <h2>Clause Analysis</h2>

      <h3>Detected Clauses</h3>
      <div className="clause-grid">
        {detected_clauses.map((c) => (
          <span key={c} className={`clause-tag risk-${(clause_risks[c] || "unknown").toLowerCase()}`}>
            {c} <em>{clause_risks[c] || "unknown"}</em>
          </span>
        ))}
      </div>

      {co_occurrence.length > 0 && (
        <>
          <h3>Co-occurrence</h3>
          <table className="co-table">
            <thead><tr><th>Clause A</th><th>Clause B</th><th>Count</th></tr></thead>
            <tbody>
              {co_occurrence.sort((a, b) => b.count - a.count).map((row, i) => (
                <tr key={i}><td>{row.a}</td><td>{row.b}</td><td>{row.count}</td></tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <h3>Contract Advice</h3>
      <p>{advice_missing}</p>
      {advice_unusual && <p className="warn-text">{advice_unusual}</p>}

      {top_lawyers.length > 0 && (
        <>
          <h3>Recommended Lawyers</h3>
          {top_lawyers.map((l, i) => (
            <div key={i} className="lawyer-card">
              <strong>{l.name}</strong> — {l.specialization} ({l.experience} yrs)
            </div>
          ))}
        </>
      )}
    </section>
  );
}
