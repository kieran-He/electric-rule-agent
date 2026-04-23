"""
LangChain Integration for RAG System

This module provides LangChain-based components for the RAG system,
enabling LCEL chain flows and future Agent expansion.
"""
from app.langchain.llm import create_minimax_llm
from app.langchain.prompts import QA_PROMPT, COMPARE_PROMPT
from app.langchain.retriever_wrapper import ChromaRepositoryRetriever
from app.langchain.chains import build_qa_chain, build_compare_chain
from app.langchain.orchestrator import LangChainQAOrchestrator

__all__ = [
    "create_minimax_llm",
    "QA_PROMPT",
    "COMPARE_PROMPT",
    "ChromaRepositoryRetriever",
    "build_qa_chain",
    "build_compare_chain",
    "LangChainQAOrchestrator",
]