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

