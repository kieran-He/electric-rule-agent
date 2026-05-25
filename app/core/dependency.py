from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.db.session import SessionLocal, init_db
from app.services.conversation_service import ConversationService
from app.services.feedback_service import FeedbackService
from app.services.dialog_manager import DialogManager
from app.services.ingest_service import IngestService
from app.services.query_service import QueryService
from app.services.trace_service import TraceService
from app.services.benchmark_service import BenchmarkService
from app.services.title_generator import TitleGenerator
from app.agent.agent_singleton import agent_singleton


@lru_cache
def get_query_service() -> QueryService:
    init_db()
    return QueryService(settings=settings, session_factory=SessionLocal)


@lru_cache
def get_ingest_service() -> IngestService:
    init_db()
    return IngestService(settings=settings, session_factory=SessionLocal)


@lru_cache
def get_trace_service() -> TraceService:
    init_db()
    return TraceService(session_factory=SessionLocal)


@lru_cache
def get_conversation_service() -> ConversationService:
    init_db()
    return ConversationService(session_factory=SessionLocal)


@lru_cache
def get_feedback_service() -> FeedbackService:
    init_db()
    return FeedbackService(session_factory=SessionLocal)


@lru_cache
def get_dialog_manager() -> DialogManager:
    init_db()
    return DialogManager(session_factory=SessionLocal)


@lru_cache
def get_benchmark_service() -> BenchmarkService:
    return BenchmarkService()


@lru_cache
def get_title_generator() -> TitleGenerator:
    llm_wrapper = None
    try:
        agent = agent_singleton.get_agent(SessionLocal())
        llm_wrapper = getattr(agent, '_llm_wrapper', None)
        if llm_wrapper is None:
            llm_wrapper = getattr(agent, 'llm_wrapper', None)
    except Exception:
        pass
    return TitleGenerator(llm_wrapper=llm_wrapper)
