import { motion } from "framer-motion";
import { List, Loader2, Target, AlertTriangle, UserCheck } from "lucide-react";

export default function ClausesPage({ analysis, handleAnalyze, loading, embedded, error }) {
  if (!embedded) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center text-gray-400 mb-4">
          <List size={32} />
        </div>
        <h2 className="text-xl font-medium text-gray-200">No Document Embedded</h2>
        <p className="text-gray-500 text-sm mt-2">Upload and embed a document first to analyze clauses.</p>
      </div>
    );
  }

  const getRiskColor = (risk) => {
    switch (risk?.toLowerCase()) {
      case "high": return "bg-red-500/10 text-red-400 border border-red-500/20";
      case "medium": return "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20";
      case "low": return "bg-green-500/10 text-green-400 border border-green-500/20";
      default: return "bg-gray-500/10 text-gray-400 border border-gray-500/20";
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="max-w-5xl mx-auto pb-10"
    >
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Clause Analysis</h1>
          <p className="text-gray-400 text-sm">Deep insights into document clauses, risk factors, and legal advice.</p>
        </div>
        <button
          onClick={handleAnalyze}
          disabled={!!loading}
          className="bg-accent-purple hover:bg-purple-600 text-white px-5 py-2.5 rounded-xl text-sm font-medium transition-all flex items-center gap-2 disabled:opacity-50"
        >
          {loading ? (
            <><Loader2 size={16} className="animate-spin" /> Analyzing...</>
          ) : (
            <><Target size={16} /> Run Analysis</>
          )}
        </button>
      </div>

      {error && (
        <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
          {error}
        </div>
      )}

      {analysis ? (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          {/* Detected Clauses */}
          <section className="glass-panel rounded-2xl p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <List size={20} className="text-accent-blue" /> Detected Clauses
            </h2>
            <div className="flex flex-wrap gap-3">
              {analysis.detected_clauses.map((c) => (
                <div key={c} className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all hover:scale-105 cursor-default ${getRiskColor(analysis.clause_risks[c])}`}>
                  {c} <span className="opacity-50 ml-1 font-normal capitalize">({analysis.clause_risks[c] || "unknown"})</span>
                </div>
              ))}
            </div>
          </section>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Advice */}
            <section className="glass-panel rounded-2xl p-6 flex flex-col gap-4">
              <div>
                <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <AlertTriangle size={20} className="text-yellow-400" /> Missing Clauses
                </h2>
                <div className="p-4 rounded-xl bg-white/5 border border-white/5 text-sm text-gray-300">
                  {analysis.advice_missing || "No missing clauses detected."}
                </div>
              </div>

              {analysis.advice_unusual && (
                <div>
                  <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                    <AlertTriangle size={20} className="text-red-400" /> Unusual Phrases
                  </h2>
                  <div className="p-4 rounded-xl bg-red-500/5 border border-red-500/10 text-sm text-red-300/80">
                    {analysis.advice_unusual}
                  </div>
                </div>
              )}
            </section>

             {/* Lawyers */}
             {analysis.top_lawyers?.length > 0 && (
              <section className="glass-panel rounded-2xl p-6">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <UserCheck size={20} className="text-accent-purple" /> Recommended Lawyers
                </h2>
                <div className="space-y-3">
                  {analysis.top_lawyers.map((l, i) => (
                    <div key={i} className="flex items-center justify-between p-4 rounded-xl bg-white/[0.03] border border-white/[0.06] hover:bg-white/[0.06] transition">
                      <div>
                        <p className="font-semibold text-gray-200">{l.name}</p>
                        <p className="text-xs text-gray-400">{l.specialization}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium text-accent-blue">{l.experience} yrs</p>
                        <p className="text-xs text-gray-500">experience</p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>

          {/* Co-occurrence */}
          {analysis.co_occurrence?.length > 0 && (
            <section className="glass-panel rounded-2xl p-6 overflow-hidden">
              <h2 className="text-lg font-semibold mb-4">Co-occurrence Analysis</h2>
              <div className="overflow-x-auto rounded-xl border border-white/5">
                <table className="w-full text-left text-sm text-gray-300">
                  <thead className="bg-white/[0.03] text-gray-500">
                    <tr>
                      <th className="px-4 py-3 font-medium">Clause A</th>
                      <th className="px-4 py-3 font-medium">Clause B</th>
                      <th className="px-4 py-3 font-medium w-24 text-center">Count</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.04] bg-black/20">
                    {analysis.co_occurrence.sort((a, b) => b.count - a.count).map((row, i) => (
                      <tr key={i} className="hover:bg-white/5 transition-colors">
                        <td className="px-4 py-3">{row.a}</td>
                        <td className="px-4 py-3">{row.b}</td>
                        <td className="px-4 py-3 text-center">
                          <span className="bg-accent-blue/20 text-accent-blue px-2 py-1 rounded-md text-xs font-bold">{row.count}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

        </motion.div>
      ) : (
        !loading && (
          <div className="glass-panel rounded-2xl p-12 text-center border border-dashed border-white/10 mt-6">
            <Target size={32} className="mx-auto text-gray-500 mb-4" />
            <p className="text-gray-400">Click analyze to process document clauses.</p>
          </div>
        )
      )}
    </motion.div>
  );
}
