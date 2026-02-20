from rag_tool.rag_engine import RAGEngine

rag_engine = RAGEngine()

def ehr_rag_tool(question: str) -> str:
    """
    Tool: Answers questions from EHR PDF.
    """
    return rag_engine.query(question)