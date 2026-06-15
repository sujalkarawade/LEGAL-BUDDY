from langchain_core.prompts import ChatPromptTemplate

QA_PROMPT = ChatPromptTemplate.from_template(
    """
You are a legal document assistant.
- Summarize content in simple language.
- Identify potential risks.
- Explain complex clauses.
Answer only from the provided context.
<context>
{context}
</context>
Question: {input}
"""
)

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """
Summarize this legal document in simple, everyday language.
Cover the important legal terms and clauses while summarizing.
Explain each legal clause in simple terms and highlight any risks mentioned in the document.
<context>
{context}
</context>
"""
)

MITIGATE_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert legal drafter and risk analyst.
Analyze the following document excerpt to find the text related to the {clause_type} clause.
Provide your response in JSON format with exactly these three keys:
- "original_text": The exact excerpted text of the clause as found in the context.
- "explanation": A brief explanation of the risks associated with this clause as written.
- "mitigated_text": A newly drafted, safer version of this clause that mitigates the risk and is fair to both parties.

<context>
{context}
</context>
"""
)

