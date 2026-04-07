import { useState } from "react";
import { askQuestion } from "../api";

export default function QAPanel() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleAsk(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    try {
      const data = await askQuestion(question);
      setAnswer(data.answer);
      setChunks(data.chunks || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>Ask a Question</h2>
      <form onSubmit={handleAsk} className="qa-form">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about the document..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Thinking..." : "Ask"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {answer && (
        <>
          <h3>Answer</h3>
          <p>{answer}</p>
          {chunks.length > 0 && (
            <details>
              <summary>Relevant Document Chunks ({chunks.length})</summary>
              {chunks.map((c, i) => (
                <div key={i} className="chunk">
                  <span className="source">Source: {c.source}</span>
                  <p>{c.content}</p>
                </div>
              ))}
            </details>
          )}
        </>
      )}
    </section>
  );
}
