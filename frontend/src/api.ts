const BASE = "http://localhost:8000/api";



export interface UploadResponse {
  filename: string;
}

export interface EmbedResponse {
  backend: string;
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



export interface GenerateDocumentPayload {
  docType: string;
  partyA: string;
  partyB: string;
  context: string;
  jurisdiction: string;
  tone: string;
  specialClauses: string;
  duration?: string;
  scope?: string;
  propertyAddress?: string;
  rent?: string;
  salary?: string;
  startDate?: string;
}

export interface GenerateDocumentResponse {
  document_text: string;
}

export async function generateDocument(payload: GenerateDocumentPayload): Promise<GenerateDocumentResponse> {
  const res = await fetch(`${BASE}/generate/document`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error((await res.json()).detail ?? "Generation failed");
  return res.json();
}
