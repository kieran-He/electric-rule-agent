from __future__ import annotations

import logging
from typing import Callable

from sqlalchemy.orm import Session

from app.schemas.answer import QueryAnswer
from app.schemas.query import QueryRequest
from app.services.conversation_service import ConversationService
from app.services.qa.orchestrator import QAOrchestrator
from app.services.trace_service import TraceService

logger = logging.getLogger(__name__)


class QueryService:
    def __init__(self, settings, session_factory: Callable[[], Session]):
        self.settings = settings
        self.session_factory = session_factory
        self._langchain_orchestrator = None
        self.conversation_service = ConversationService(session_factory)

    def _get_orchestrator(self, db: Session):
        """Get orchestrator based on USE_LANGCHAIN and USE_HYBRID_RETRIEVAL flags."""
        if self.settings.use_langchain:
            if self.settings.use_hybrid_retrieval:
                from app.langchain.orchestrator_hybrid import HybridQAOrchestrator
                return HybridQAOrchestrator(db=db, settings=self.settings)
            else:
                from app.langchain.orchestrator import LangChainQAOrchestrator
                return LangChainQAOrchestrator(db=db, settings=self.settings)
        else:
            return QAOrchestrator(db=db, settings=self.settings)

    def answer(self, req: QueryRequest) -> QueryAnswer:
        history = self.conversation_service.get_history(req.session_id)
        
        with self.session_factory() as db:
            trace_service = TraceService(self.session_factory)
            orchestrator = self._get_orchestrator(db)
            
            if self.settings.use_langchain:
                if self.settings.use_hybrid_retrieval:
                    response = orchestrator.run(req, history=history, trace_service=trace_service)
                else:
                    response = orchestrator.run(req, history=history)
            else:
                response = orchestrator.run(req)
            
            self.conversation_service.append_turn(
                session_id=req.session_id,
                user_query=req.query,
                bot_reply=response.answer,
                intent=response.intent
            )
            
            db.commit()
            return response
