const BASE = "http://localhost:8000/api";

export interface StatusResponse {
  groq: boolean;
  openrouter: boolean;
}

export interface UploadResponse {
  filename: string;
}

export interface EmbedResponse {
  backend: string;
}

export interface SummarizeResponse {
  summary: string;
}

export interface AskResponse {
  answer: string;
  chunks?: Array<{
    source: string;
    content: string;
  }>;
}

export interface AnalysisResponse {
  detected_clauses: string[];
  clause_risks: Record<string, string>;
  advice_missing: string;
  advice_unusual?: string;
  top_lawyers?: Array<{
    name: string;
    specialization: string;
    experience: number;
  }>;
  co_occurrence?: Array<{
    a: string;
    b: string;
    count: number;
  }>;
}

export async function getStatus(): Promise<StatusResponse> {
  const res = await fetch(`${BASE}/status`);
  return res.json();
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/documents/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}

export async function embedDocument(filename: string): Promise<EmbedResponse> {
  const res = await fetch(`${BASE}/documents/embed?filename=${encodeURIComponent(filename)}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}

export async function summarize(): Promise<SummarizeResponse> {
  const res = await fetch(`${BASE}/qa/summarize`, { method: "POST" });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const res = await fetch(`${BASE}/qa/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}

export async function getAnalysis(): Promise<AnalysisResponse> {
  const res = await fetch(`${BASE}/analysis/clauses`);
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}

export interface MitigateResponse {
  original_text: string;
  explanation: string;
  mitigated_text: string;
}

export async function mitigateClause(clause: string): Promise<MitigateResponse> {
  const res = await fetch(`${BASE}/analysis/mitigate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clause }),
  });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}
