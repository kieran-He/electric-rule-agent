"""
LCEL Chain Construction for RAG QA System

Builds LangChain Expression Language (LCEL) chains for QA and comparison tasks.
"""
from typing import List, Optional

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from app.langchain.llm import create_minimax_llm
from app.langchain.prompts import QA_PROMPT, COMPARE_PROMPT
from app.langchain.retriever_wrapper import ChromaRepositoryRetriever, format_chunks_for_context
from app.core.repository import PolicyChunk


def build_qa_chain(
    llm=None,
    retriever=None,
    province_code: str = "SN",
):
    """
    Build QA chain using LCEL.

    Chain flow:
    1. Retrieve context from ChromaPolicyRepository
    2. Format context as string
    3. Apply QA prompt template
    4. Invoke LLM
    5. Parse output as string

    Args:
        llm: LangChain LLM instance (defaults to MiniMax)
        retriever: Retriever instance
        province_code: Default province code for context

    Returns:
        LCEL Runnable chain
    """
    if llm is None:
        llm = create_minimax_llm()

    def format_provincial_context(chunks: List[PolicyChunk]) -> str:
        return format_chunks_for_context(chunks)

    def empty_history(_: dict) -> str:
        return ""

    chain = (
        {
            "question": RunnablePassthrough(),
            "provincial_context": retriever | RunnableLambda(format_provincial_context),
            "global_context": RunnableLambda(lambda _: "- 无通用证据"),
            "province_code": RunnableLambda(lambda _: province_code),
            "history": RunnableLambda(empty_history),
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def build_compare_chain(
    llm=None,
    retriever_by_province: Optional[dict] = None,
):
    """
    Build cross-province comparison chain using LCEL.

    Args:
        llm: LangChain LLM instance
        retriever_by_province: Dict mapping province codes to retrievers

    Returns:
        LCEL Runnable chain for comparison
    """
    if llm is None:
        llm = create_minimax_llm()

    def format_cross_province_context(input_data: dict) -> str:
        query = input_data.get("question", "")
        province_chunks = input_data.get("province_chunks", {})

        if not province_chunks:
            return "- 无跨省证据"

        lines = []
        for province, chunks in province_chunks.items():
            lines.append(f"\n=== {province} ===")
            lines.append(format_chunks_for_context(chunks))

        return "\n".join(lines)

    chain = (
        {
            "question": RunnablePassthrough(),
            "cross_province_context": RunnableLambda(format_cross_province_context),
        }
        | COMPARE_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


class QAChainRunner:
    """
    Runner for QA chain with context retrieval.

    Handles the full flow: retrieve -> chain invoke -> build citations.
    """

    def __init__(
        self,
        llm=None,
        retriever: ChromaRepositoryRetriever = None,
        province_code: str = "SN",
    ):
        self.llm = llm or create_minimax_llm()
        self.retriever = retriever
        self.province_code = province_code
        self._chain = None

    def _build_chain(self):
        if self._chain is None:
            self._chain = build_qa_chain(
                llm=self.llm,
                retriever=self.retriever,
                province_code=self.province_code,
            )
        return self._chain

    def run(self, query: str) -> tuple[str, List[PolicyChunk]]:
        """
        Run QA chain and return answer with chunks.

        Args:
            query: User query

        Returns:
            Tuple of (answer_text, chunks_used)
        """
        chain = self._build_chain()

        # First retrieve chunks for citations
        chunks = self.retriever.invoke(query) if self.retriever else []

        # Invoke chain
        answer = chain.invoke(query)

        return answer, chunks