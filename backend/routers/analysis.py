import itertools
import json
import re
from collections import Counter

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import backend.state as state
from app.analysis import (
    advice_on_missing_clauses,
    advice_on_unusual_combinations,
    get_all_detected_clauses,
    get_clause_risk,
    get_top_lawyers,
)
from app.config import LAWYER_PATH
from app.data import load_json_list, load_risk_data
from app.llm import invoke_with_groq_fallback
from app.prompts import MITIGATE_PROMPT

router = APIRouter()

class MitigateRequest(BaseModel):
    clause: str


@router.get("/clauses")
def clause_analysis():
    if not state.final_docs:
        raise HTTPException(status_code=400, detail="Embed a document first.")

    all_detected: set[str] = set()
    for doc in state.final_docs:
        all_detected.update(doc.metadata.get("clauses", []))

    all_clauses_list = get_all_detected_clauses(state.final_docs)
    pairs = []
    for clauses in all_clauses_list:
        pairs.extend(itertools.combinations(sorted(set(clauses)), 2))
    co_occurrence = [{"a": a, "b": b, "count": count} for (a, b), count in Counter(pairs).items()]

    risk_df = load_risk_data()
    clause_risks = {clause: get_clause_risk(clause, risk_df) for clause in all_detected}

    advice_missing = advice_on_missing_clauses(all_detected)
    advice_unusual = advice_on_unusual_combinations(all_detected)

    lawyer_rag = load_json_list(LAWYER_PATH)
    top_lawyers = get_top_lawyers(lawyer_rag) if "HIGH" in clause_risks.values() else []

    return {
        "detected_clauses": list(all_detected),
        "clause_risks": clause_risks,
        "co_occurrence": co_occurrence,
        "advice_missing": advice_missing,
        "advice_unusual": advice_unusual,
        "top_lawyers": top_lawyers,
    }


@router.post("/mitigate")
def mitigate_clause(req: MitigateRequest):
    if not state.final_docs:
        raise HTTPException(status_code=400, detail="Embed a document first.")

    target_content = ""
    for doc in state.final_docs:
        if doc.metadata.get("source") == "PDF" and req.clause in doc.metadata.get("clauses", []):
            target_content = doc.page_content
            break

    if not target_content:
        raise HTTPException(status_code=404, detail="Clause text not found in the uploaded document.")

    prompt_val = MITIGATE_PROMPT.format_messages(clause_type=req.clause, context=target_content)

    def factory(llm):
        return llm.invoke(prompt_val).content

    try:
        raw_output = invoke_with_groq_fallback(factory)
        
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the model's response.")
        
        parsed = json.loads(match.group(0))
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mitigate clause: {str(e)}")
