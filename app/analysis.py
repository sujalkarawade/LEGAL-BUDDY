import itertools
import re
from collections import Counter

import pandas as pd
import streamlit as st
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
    clauses = []
    for pattern, label in CLAUSE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            clauses.append(label)
    return clauses


def build_summary_docs(
    docs: list[Document],
    max_input_tokens: int = 3500,
    approx_chars_per_token: int = 4,
) -> list[Document]:
    char_limit = max_input_tokens * approx_chars_per_token
    selected: list[Document] = []
    total_chars = 0

    for doc in docs:
        if total_chars >= char_limit:
            break

        content = doc.page_content or ""
        remaining = char_limit - total_chars
        if remaining <= 0:
            break

        if len(content) <= remaining:
            selected.append(doc)
            total_chars += len(content)
        else:
            selected.append(Document(page_content=content[:remaining], metadata=doc.metadata))
            break

    return selected


def get_all_detected_clauses(docs: list[Document]) -> list[list[str]]:
    all_clauses = []
    for doc in docs:
        clauses = doc.metadata.get("clauses", [])
        if clauses:
            all_clauses.append(clauses)
    return all_clauses


def show_cooccurrence_matrix(docs: list[Document]) -> None:
    all_clauses = get_all_detected_clauses(docs)
    pairs = []
    for clauses in all_clauses:
        pairs.extend(itertools.combinations(sorted(set(clauses)), 2))
    co_occurrence = Counter(pairs)
    if co_occurrence:
        df = (
            pd.DataFrame({(a, b): count for (a, b), count in co_occurrence.items()}, index=[0])
            .T.sort_values(0, ascending=False)
        )
        st.subheader("Clause Co-Occurrence Matrix")
        st.dataframe(df)
    else:
        st.info("No clause pairs detected for co-occurrence analysis.")


def advice_on_missing_clauses(
    detected_clauses: set[str], expected_clauses: list[str] = EXPECTED_CLAUSES
) -> str:
    missing = [clause for clause in expected_clauses if clause not in detected_clauses]
    if missing:
        return (
            "Missing standard clauses: "
            + ", ".join(missing)
            + ". Consider adding them for better protection."
        )
    return "All standard clauses are present."


def advice_on_unusual_combinations(detected_clauses: set[str]) -> str:
    if "TERMINATION" in detected_clauses and "NOTICE_PERIOD" not in detected_clauses:
        return "TERMINATION exists but NOTICE_PERIOD is missing. This may weaken protection."
    return ""


def get_clause_risk(clause_type: str, risk_df: pd.DataFrame) -> str:
    if risk_df.empty:
        return "unknown"
    match = risk_df[risk_df["clause_type"].str.lower() == clause_type.lower()]
    if not match.empty:
        return str(match["risk_level"].iloc[0])
    return "unknown"


def get_top_lawyers(lawyer_rag: list[dict], area: str = "Civil", top_n: int = 3) -> list[dict]:
    filtered = [lawyer for lawyer in lawyer_rag if lawyer.get("specialization", "").lower() == area.lower()]
    filtered.sort(key=lambda lawyer: -lawyer.get("experience", 0))
    return filtered[:top_n]

