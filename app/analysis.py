import itertools
import re
from collections import Counter

import pandas as pd
from langchain_core.documents import Document

CLAUSE_PATTERNS = [
    (r"\bparties\b|\bbetween\b", "PARTIES"),
    (r"\bobject\b|\bpurpose\b", "OBJECT_PURPOSE"),
    (r"\bconsideration\b|\bpayment\b|\bfees?\b|\brent\b", "CONSIDERATION_PAYMENT"),
    (r"\bterm\b|\bduration\b|\bvalidity\b", "TERM_DURATION"),
    (r"\bobligations?\b|\bduties\b|\bresponsibilit(y|ies)\b", "OBLIGATIONS"),
    (r"\bright(s)?\b|\bprivileges?\b", "RIGHTS"),
    (r"\btermination\b|\bcancellation\b|\brescind\b", "TERMINATION"),
    (r"\bliabilit(y|ies)\b|\bindemnif(y|ication)\b", "LIABILITY_INDEMNITY"),
    (r"\bconfidential(ity)?\b|\bsecrecy\b", "CONFIDENTIALITY"),
    (r"\bdispute\b|\barbitration\b|\bjurisdiction\b", "DISPUTE_RESOLUTION"),
    (r"\bgoverning\s+law\b|\bapplicable\s+law\b", "GOVERNING_LAW"),
    (r"\bforce\s+majeure\b|\bact\s+of\s+god\b", "FORCE_MAJEURE"),
    (r"\bnotice\b|\bcommunication\b", "NOTICE_COMMUNICATION"),
    (r"\bsign(ed|ature)?\b|\bexecution\b", "SIGNATURE_EXECUTION"),
]

EXPECTED_CLAUSES = ["TERM_DURATION", "RENT_PAYMENT", "SECURITY_DEPOSIT", "TERMINATION"]


def identify_clauses(text: str) -> list[str]:
    return [label for pattern, label in CLAUSE_PATTERNS if re.search(pattern, text, re.IGNORECASE)]


def build_summary_docs(docs: list[Document], max_input_tokens: int = 3500,
                       approx_chars_per_token: int = 4) -> list[Document]:
    char_limit = max_input_tokens * approx_chars_per_token
    selected: list[Document] = []
    total_chars = 0
    for doc in docs:
        if total_chars >= char_limit:
            break
        content = doc.page_content or ""
        remaining = char_limit - total_chars
        if len(content) <= remaining:
            selected.append(doc)
            total_chars += len(content)
        else:
            selected.append(Document(page_content=content[:remaining], metadata=doc.metadata))
            break
    return selected


def get_all_detected_clauses(docs: list[Document]) -> list[list[str]]:
    return [doc.metadata.get("clauses", []) for doc in docs if doc.metadata.get("clauses")]


def get_cooccurrence(docs: list[Document]) -> list[dict]:
    pairs = []
    for clauses in get_all_detected_clauses(docs):
        pairs.extend(itertools.combinations(sorted(set(clauses)), 2))
    return [{"a": a, "b": b, "count": c} for (a, b), c in Counter(pairs).items()]


def advice_on_missing_clauses(detected: set[str], expected: list[str] = EXPECTED_CLAUSES) -> str:
    missing = [c for c in expected if c not in detected]
    if missing:
        return "Missing standard clauses: " + ", ".join(missing) + ". Consider adding them for better protection."
    return "All standard clauses are present."


def advice_on_unusual_combinations(detected: set[str]) -> str:
    if "TERMINATION" in detected and "NOTICE_PERIOD" not in detected:
        return "TERMINATION exists but NOTICE_PERIOD is missing. This may weaken protection."
    return ""


def get_clause_risk(clause_type: str, risk_df: pd.DataFrame) -> str:
    if risk_df.empty:
        return "unknown"
    match = risk_df[risk_df["clause_type"].str.lower() == clause_type.lower()]
    return str(match["risk_level"].iloc[0]) if not match.empty else "unknown"


def get_top_lawyers(lawyer_rag: list[dict], area: str = "Civil", top_n: int = 3) -> list[dict]:
    filtered = [l for l in lawyer_rag if l.get("specialization", "").lower() == area.lower()]
    filtered.sort(key=lambda l: -l.get("experience", 0))
    return filtered[:top_n]
