import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# ── Document type → full name mapping ──────────────────────────
DOC_TYPE_LABELS = {
    "nda":               "Non-Disclosure Agreement (NDA)",
    "employment":        "Employment Contract",
    "lease":             "Lease / Rental Agreement",
    "service":           "Service Agreement",
    "partnership":       "Partnership Deed",
    "power_of_attorney": "Power of Attorney",
    "affidavit":         "Affidavit",
    "legal_notice":      "Legal Notice",
    "custom":            "Custom Legal Document",
}

TONE_LABELS = {
    "formal":     "formal and standard legal language",
    "semi_formal": "semi-formal, balanced language",
    "plain":      "plain language that is easy to read and understand",
}


class GenerateRequest(BaseModel):
    docType: str
    partyA: str
    partyB: Optional[str] = ""
    context: str
    jurisdiction: str = "India"
    tone: str = "formal"
    specialClauses: Optional[str] = ""
    # type-specific fields
    duration: Optional[str] = None
    scope: Optional[str] = None
    propertyAddress: Optional[str] = None
    rent: Optional[str] = None
    salary: Optional[str] = None
    startDate: Optional[str] = None


def build_prompt(req: GenerateRequest) -> str:
    doc_label  = DOC_TYPE_LABELS.get(req.docType, req.docType)
    tone_label = TONE_LABELS.get(req.tone, "formal legal language")

    extra_fields = []
    if req.duration:        extra_fields.append(f"- Confidentiality / Duration: {req.duration}")
    if req.scope:           extra_fields.append(f"- Scope: {req.scope}")
    if req.propertyAddress: extra_fields.append(f"- Property Address: {req.propertyAddress}")
    if req.rent:            extra_fields.append(f"- Monthly Rent: {req.rent}")
    if req.salary:          extra_fields.append(f"- Salary / CTC: {req.salary}")
    if req.startDate:       extra_fields.append(f"- Start Date: {req.startDate}")
    if req.specialClauses:  extra_fields.append(f"- Special Clauses / Instructions: {req.specialClauses}")

    extra_block = ("\n" + "\n".join(extra_fields)) if extra_fields else ""

    prompt = f"""You are an expert legal document drafter. Generate a complete, professional {doc_label} based on the following details.

DOCUMENT DETAILS:
- Document Type: {doc_label}
- Jurisdiction: {req.jurisdiction}
- Party A: {req.partyA}
- Party B: {req.partyB if req.partyB else "N/A"}
- Purpose / Context: {req.context}
- Tone: {tone_label}{extra_block}

INSTRUCTIONS:
1. Draft the full {doc_label} with all standard clauses and sections.
2. Use {tone_label}.
3. Apply laws and conventions applicable in {req.jurisdiction}.
4. Include: title, preamble, definitions, all relevant clauses, signatures section, date placeholders.
5. Make it complete and ready to use (add [DATE], [PLACE] placeholders where needed).
6. Do NOT add commentary or explanations — output only the document text.

Generate the complete document now:"""

    return prompt


@router.post("/document")
async def generate_document(req: GenerateRequest):
    """Generate a legal document using GROQ LLM."""
    try:
        from groq import Groq
    except ImportError:
        raise HTTPException(status_code=500, detail="Groq library not installed. Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured on the server.")

    prompt = build_prompt(req)

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert legal document drafter with deep knowledge of international law. "
                        "You produce professional, complete legal documents formatted in plain text. "
                        "Never include disclaimers or meta-commentary in your output."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        document_text = response.choices[0].message.content.strip()
        return {"document_text": document_text}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document generation failed: {str(exc)}")
