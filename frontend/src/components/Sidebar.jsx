export default function Sidebar({ status, embedded, embedBackend }) {
  return (
    <aside className="sidebar">
      <h2>Status</h2>
      <div className={`badge ${embedded ? "ok" : "warn"}`}>
        {embedded ? "Vector DB ready" : "Vector DB not initialized"}
      </div>
      {embedBackend && <p className="caption">Embeddings: {embedBackend}</p>}
      <div className={`badge ${status.groq ? "ok" : "warn"}`}>
        {status.groq ? "Groq API detected" : "GROQ_API_KEY missing"}
      </div>
      <div className={`badge ${status.openrouter ? "ok" : "info"}`}>
        {status.openrouter ? "OpenRouter API detected" : "OpenRouter not configured"}
      </div>
    </aside>
  );
}
