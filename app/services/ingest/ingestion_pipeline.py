from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models.clause import Clause
from app.db.models.document import Document
from app.db.models.rule_tag import RuleTag
from app.core.repository import ChromaPolicyRepository


class IngestionPipeline:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.repo = ChromaPolicyRepository(
            persist_directory=settings.chroma_path,
            embedding_model_name=settings.embedding_model,
        )

    def ingest_path(self, path: Path, province_code: str, rebuild_index: bool) -> dict[str, int]:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file() and path.suffix.lower() == ".json":
            return self._ingest_processed_json(path, province_code, rebuild_index)

        if path.is_dir():
            return self._ingest_directory(path, province_code, rebuild_index)

        raise ValueError(f"Unsupported path type: {path}")

    def _ingest_processed_json(self, json_path: Path, province_code: str, rebuild_index: bool) -> dict[str, int]:
        file_hash = json_path.stem
        existing = self.db.scalar(
            self.db.query(Document).where(Document.file_hash == file_hash).statement
        )
        if existing:
            return {"imported_documents": 0, "imported_clauses": 0, "skipped_documents": 1}

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        doc = Document(
            province_code=meta.get("province_code", province_code),
            doc_name=meta.get("doc_name", ""),
            doc_type=meta.get("doc_type", "notice"),
            market_type=meta.get("market_type"),
            subject_scope=json.dumps(meta.get("subject_scope", [])) if meta.get("subject_scope") else None,
            version_name=meta.get("version_name"),
            status=meta.get("status", "formal"),
            issue_date=meta.get("issue_date"),
            effective_date=meta.get("effective_date"),
            source_file=meta.get("source_file", str(json_path)),
            file_hash=file_hash,
            is_current=meta.get("is_current", True),
            parent_doc_id=None,
            raw_text=data.get("cleaned_text"),
        )
        self.db.add(doc)
        self.db.flush()

        clauses_data = data.get("clauses", [])
        clauses: list[Clause] = []
        texts: list[str] = []
        metadatas: list[dict[str, str]] = []
        rule_tags: list[RuleTag] = []

        for clause_data in clauses_data:
            clause = Clause(
                doc_id=doc.id,
                chapter_no=clause_data.get("chapter_no"),
                chapter_title=clause_data.get("chapter_title"),
                section_no=clause_data.get("section_no"),
                section_title=clause_data.get("section_title"),
                article_no=clause_data.get("article_no"),
                item_no=clause_data.get("item_no"),
                title_path=clause_data.get("title_path", ""),
                clause_text=clause_data.get("clause_text", ""),
                clause_summary=clause_data.get("clause_summary"),
                page_start=clause_data.get("page_start"),
                page_end=clause_data.get("page_end"),
                token_count=clause_data.get("token_count", 0),
            )
            clauses.append(clause)
            texts.append(clause_data.get("clause_text", ""))
            metadatas.append(self._build_metadata(clause_data, doc))

        self.db.add_all(clauses)
        self.db.flush()

        for i, clause_data in enumerate(clauses_data):
            tags = clause_data.get("rule_tags", {})
            if tags:
                rule_tag = RuleTag(
                    clause_id=clauses[i].id,
                    province_code=doc.province_code,
                    market_type=tags.get("market_type"),
                    entity_type=tags.get("entity_type"),
                    trade_cycle=tags.get("trade_cycle"),
                    trade_mode=tags.get("trade_mode"),
                    time_granularity=tags.get("time_granularity"),
                    action_type=tags.get("action_type"),
                    penalty_related=tags.get("penalty_related", False),
                    green_power_related=tags.get("green_power_related", False),
                    spot_related=tags.get("spot_related", False),
                    retail_related=tags.get("retail_related", False),
                    storage_related=tags.get("storage_related", False),
                    vpp_related=tags.get("vpp_related", False),
                )
                rule_tags.append(rule_tag)

        if rule_tags:
            self.db.add_all(rule_tags)

        if texts:
            self.repo.ingest_chunks(
                texts=texts,
                metadatas=metadatas,
                kb_scope="province",
                province_code=province_code,
                rebuild=rebuild_index,
            )

        return {"imported_documents": 1, "imported_clauses": len(clauses), "skipped_documents": 0}

    def _ingest_directory(self, dir_path: Path, province_code: str, rebuild_index: bool) -> dict[str, int]:
        json_files = [f for f in dir_path.glob("*.json") if not f.name.startswith("_")]
        total_docs = 0
        total_clauses = 0
        total_skipped = 0

        all_texts: list[str] = []
        all_metadatas: list[dict[str, str]] = []

        for json_file in json_files:
            result = self._ingest_processed_json(json_file, province_code, rebuild_index)
            total_docs += result["imported_documents"]
            total_clauses += result["imported_clauses"]
            total_skipped += result["skipped_documents"]

        if all_texts:
            self.repo.ingest_chunks(
                texts=all_texts,
                metadatas=all_metadatas,
                kb_scope="province",
                province_code=province_code,
                rebuild=rebuild_index,
            )

        return {
            "imported_documents": total_docs,
            "imported_clauses": total_clauses,
            "skipped_documents": total_skipped,
        }

    def _build_metadata(self, clause_data: dict[str, Any], doc: Document) -> dict[str, str]:
        return {
            "province_code": doc.province_code,
            "doc_id": str(doc.id),
            "source_name": doc.doc_name[:200] if doc.doc_name else "",
            "source_path": doc.source_file,
            "file_hash": doc.file_hash,
            "doc_title": doc.doc_name[:200] if doc.doc_name else "",
            "effective_date": str(doc.effective_date or ""),
            "policy_level": doc.status,
            "doc_type": doc.doc_type,
            "article_no": clause_data.get("article_no", ""),
            "title_path": clause_data.get("title_path", "")[:500] if clause_data.get("title_path") else "",
        }

    def rebuild_vector_index(self) -> int:
        clauses = self.db.query(Clause).join(Document).all()
        texts: list[str] = []
        metadatas: list[dict[str, str]] = []

        for clause in clauses:
            texts.append(clause.clause_text)
            metadatas.append(self._build_metadata_from_clause(clause))

        if texts:
            self.repo.ingest_chunks(
                texts=texts,
                metadatas=metadatas,
                kb_scope="province",
                province_code=self.settings.province_default,
                rebuild=True,
            )

        return len(texts)

    def _build_metadata_from_clause(self, clause: Clause) -> dict[str, str]:
        doc = clause.document
        return {
            "province_code": doc.province_code,
            "doc_id": str(doc.id),
            "source_name": doc.doc_name[:200] if doc.doc_name else "",
            "source_path": doc.source_file,
            "file_hash": doc.file_hash,
            "doc_title": doc.doc_name[:200] if doc.doc_name else "",
            "effective_date": str(doc.effective_date or ""),
            "policy_level": doc.status,
            "doc_type": doc.doc_type,
            "article_no": clause.article_no or "",
            "title_path": clause.title_path[:500] if clause.title_path else "",
        }