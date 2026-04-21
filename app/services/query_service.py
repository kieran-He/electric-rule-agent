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

    def answer(self, req: QueryRequest) -> QueryAnswer:
        with self.session_factory() as db:
            orchestrator = QAOrchestrator(db=db, settings=self.settings)
            response = orchestrator.run(req)
            db.commit()
            return response
