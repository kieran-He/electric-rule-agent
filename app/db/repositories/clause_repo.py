from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, joinedload

from app.db.models.clause import Clause
from app.db.models.document import Document
from app.db.models.rule_tag import RuleTag


class ClauseRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_all(self, clauses: list[Clause]) -> None:
        self.db.add_all(clauses)
        self.db.flush()

    def search_keyword(self, query: str, limit: int = 10) -> list[Clause]:
        terms = [term.strip() for term in query.split() if term.strip()]
        stmt = select(Clause).options(joinedload(Clause.document)).join(Document)
        if terms:
            filters = [
                or_(
                    Clause.clause_text.contains(term),
                    Clause.title_path.contains(term),
                    Document.doc_name.contains(term),
                )
                for term in terms
            ]
            stmt = stmt.where(or_(*filters))
        stmt = stmt.limit(limit)
        return list(self.db.scalars(stmt).unique())

    def filter_by_tags(
        self,
        province_code: str | None = None,
        market_type: str | None = None,
        entity_type: str | None = None,
        status: str | None = None,
        doc_type: str | None = None,
        is_current: bool | None = None,
        limit: int = 10,
    ) -> list[Clause]:
        stmt = (
            select(Clause)
            .options(joinedload(Clause.document), joinedload(Clause.rule_tags))
            .join(Document)
            .join(RuleTag, RuleTag.clause_id == Clause.id, isouter=True)
        )
        if province_code:
            stmt = stmt.where(Document.province_code == province_code)
        if market_type:
            stmt = stmt.where(or_(Document.market_type == market_type, RuleTag.market_type == market_type))
        if entity_type:
            stmt = stmt.where(RuleTag.entity_type == entity_type)
        if status:
            stmt = stmt.where(Document.status == status)
        if doc_type:
            stmt = stmt.where(Document.doc_type == doc_type)
        if is_current is not None:
            stmt = stmt.where(Document.is_current == is_current)
        stmt = stmt.limit(limit)
        return list(self.db.scalars(stmt).unique())
