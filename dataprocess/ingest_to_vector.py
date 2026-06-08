from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataprocess.config import DocProcSettings, settings
from dataprocess.schemas import ClauseChunk, ProcessedDocument


class ChromaIngestError(RuntimeError):
    pass


def ingest_processed_document_to_chroma(
    document: ProcessedDocument,
    *,
    chroma_path: str | None = None,
    embedding_model: str | None = None,
    province_code: str | None = None,
    config: DocProcSettings | None = None,
) -> dict[str, int]:
    cfg = config or settings
    persist_dir = chroma_path or cfg.chroma_path
    model_name = embedding_model or cfg.embedding_model
    prov_code = province_code or document.metadata.province_code
    
    try:
        import chromadb
    except ImportError as exc:
        raise ChromaIngestError(f"chromadb not installed: {exc}") from exc
    
    client = chromadb.PersistentClient(path=persist_dir)
    collection_name = f"kb_{prov_code.lower()}"
    collection = client.get_or_create_collection(name=collection_name)
    
    texts: list[str] = []
    metadatas: list[dict[str, str]] = []
    
    for clause in document.clauses:
        texts.append(clause.clause_text)
        metadatas.append(_build_metadata(clause, document))
    
    if not texts:
        return {"ingested_chunks": 0}
    
    embeddings = _embed_texts(texts, model_name)
    
    ids = [f"{document.metadata.file_hash}_{i}" for i in range(len(texts))]
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    
    return {"ingested_chunks": len(texts)}


def _build_metadata(clause: ClauseChunk, document: ProcessedDocument) -> dict[str, str]:
    return {
        "province_code": document.metadata.province_code,
        "doc_name": document.metadata.doc_name[:200] if document.metadata.doc_name else "",
        "source_name": document.metadata.doc_name[:200] if document.metadata.doc_name else "",
        "source_path": document.metadata.source_file,
        "file_hash": document.metadata.file_hash,
        "doc_title": document.metadata.doc_name[:200] if document.metadata.doc_name else "",
        "issuer": document.metadata.issuer or "",
        "issue_date": str(document.metadata.issue_date or ""),
        "effective_date": str(document.metadata.effective_date or ""),
        "policy_level": document.metadata.status,
        "doc_type": document.metadata.doc_type,
        "article_no": clause.article_no or "",
        "title_path": (clause.title_path or "")[:500],
        "page_start": str(clause.page_start or ""),
        "page_end": str(clause.page_end or ""),
        "issuer": document.metadata.issuer[:200] if document.metadata.issuer else "",
    }


def _embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    if model_name.lower() in {"deterministic", "fallback"}:
        return _deterministic_embed(texts)
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        vectors = model.encode(texts, normalize_embeddings=True)
        if hasattr(vectors, "tolist"):
            return vectors.tolist()
        return vectors
    except Exception:
        return _deterministic_embed(texts)


def _deterministic_embed(texts: list[str], dimension: int = 512) -> list[list[float]]:
    import hashlib
    import math
    
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dimension
        for token in text.split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], byteorder="little", signed=False) % dimension
            vec[idx] += 1.0
        if not any(vec) and text:
            for ch in text:
                digest = hashlib.md5(ch.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:2], byteorder="little", signed=False) % dimension
                vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        vectors.append(vec)
    return vectors


def load_and_ingest_json(
    json_path: str | Path,
    *,
    chroma_path: str | None = None,
    embedding_model: str | None = None,
    province_code: str | None = None,
    config: DocProcSettings | None = None,
) -> dict[str, int]:
    cfg = config or settings
    path = Path(json_path)
    
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    document = ProcessedDocument(**data)
    
    return ingest_processed_document_to_chroma(
        document,
        chroma_path=chroma_path,
        embedding_model=embedding_model,
        province_code=province_code,
        config=cfg,
    )