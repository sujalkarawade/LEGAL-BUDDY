const BASE = "http://localhost:8000/api";

export async function getStatus() {
  const res = await fetch(`${BASE}/status`);
  return res.json();
}

export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/documents/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}

export async function embedDocument(filename) {
  const res = await fetch(`${BASE}/documents/embed?filename=${encodeURIComponent(filename)}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}

export async function summarize() {
  const res = await fetch(`${BASE}/qa/summarize`, { method: "POST" });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}

export async function askQuestion(question) {
  const res = await fetch(`${BASE}/qa/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}

export async function getAnalysis() {
  const res = await fetch(`${BASE}/analysis/clauses`);
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}
