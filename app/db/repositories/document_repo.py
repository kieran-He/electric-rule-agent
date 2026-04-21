from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.document import Document


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_hash(self, file_hash: str) -> Document | None:
        return self.db.scalar(select(Document).where(Document.file_hash == file_hash))

    def list_documents(self) -> list[Document]:
        stmt = select(Document).order_by(Document.updated_at.desc())
        return list(self.db.scalars(stmt))

    def save(self, document: Document) -> Document:
        self.db.add(document)
        self.db.flush()
        return document
