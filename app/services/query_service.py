from __future__ import annotations

from typing import Callable

from sqlalchemy.orm import Session

from app.schemas.answer import QueryAnswer
from app.schemas.query import QueryRequest
from app.services.qa.orchestrator import QAOrchestrator


class QueryService:
    def __init__(self, settings, session_factory: Callable[[], Session]):
        self.settings = settings
        self.session_factory = session_factory
        self._langchain_orchestrator = None

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
        with self.session_factory() as db:
            orchestrator = self._get_orchestrator(db)
            response = orchestrator.run(req)
            db.commit()
            return response
