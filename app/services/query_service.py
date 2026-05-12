from __future__ import annotations

import logging
from typing import Callable

from sqlalchemy.orm import Session

from app.schemas.answer import QueryAnswer
from app.schemas.query import QueryRequest
from app.services.conversation_service import ConversationService
from app.services.orchestrator_singleton import orchestrator_singleton
from app.services.trace_service import TraceService

logger = logging.getLogger(__name__)


class QueryService:
    def __init__(self, settings, session_factory: Callable[[], Session]):
        self.settings = settings
        self.session_factory = session_factory
        self.conversation_service = ConversationService(session_factory)

    def answer(self, req: QueryRequest) -> QueryAnswer:
        history = self.conversation_service.get_history(req.session_id)
        
        with self.session_factory() as db:
            trace_service = TraceService(self.session_factory)
            orchestrator = orchestrator_singleton.get_orchestrator(db)
            
            response = orchestrator.run(req, history=history, trace_service=trace_service, db=db)
            
            self.conversation_service.append_turn(
                session_id=req.session_id,
                user_query=req.query,
                bot_reply=response.answer,
                intent=response.intent
            )
            
            db.commit()
            return response