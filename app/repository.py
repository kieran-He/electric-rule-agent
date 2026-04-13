from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4


@dataclass
class PolicyChunk:
    text: str
    score: float
    metadata: Dict[str, str]


class RepositoryError(RuntimeError):
    pass


class ChromaPolicyRepository:
    def __init__(self, persist_directory: str, embedding_model_name: str):
        self._ready = False
        self._client = None
        self._embedder = None
        self._persist_directory = Path(persist_directory)
        self._embedding_model_name = embedding_model_name

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self._persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_directory))
            self._embedder = SentenceTransformer(self._embedding_model_name)
            self._ready = True
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self._init_error = str(exc)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def init_error(self) -> Optional[str]:
        return getattr(self, "_init_error", None)

    def _collection_name(self, kb_scope: str, province_code: Optional[str]) -> str:
        if kb_scope == "global":
            return "kb_global"
        if not province_code:
            raise RepositoryError("province scope requires province_code")
        return f"kb_{province_code.lower()}"

    def _get_or_create_collection(self, name: str):
        if not self._ready:
            raise RepositoryError(self.init_error or "repository not ready")
        return self._client.get_or_create_collection(name=name)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not self._ready:
            raise RepositoryError(self.init_error or "repository not ready")
        vectors = self._embedder.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def ingest_chunks(
        self,
        texts: List[str],
        metadatas: List[Dict[str, str]],
        kb_scope: str,
        province_code: Optional[str],
        rebuild: bool = False,
    ) -> int:
        collection_name = self._collection_name(kb_scope, province_code)
        collection = self._get_or_create_collection(collection_name)
        if rebuild:
            self._client.delete_collection(collection_name)
            collection = self._get_or_create_collection(collection_name)
        ids = [str(uuid4()) for _ in texts]
        embeddings = self._embed(texts)
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        return len(ids)

    def delete_by_file_hash(
        self,
        kb_scope: str,
        province_code: Optional[str],
        file_hash: str,
    ) -> None:
        if not file_hash:
            return
        collection_name = self._collection_name(kb_scope, province_code)
        collection = self._get_or_create_collection(collection_name)
        try:
            collection.delete(where={"file_hash": file_hash})
        except Exception as exc:
            raise RepositoryError(f"failed to delete old chunks by file_hash: {exc}") from exc

    def retrieve(
        self, query: str, top_k: int, kb_scope: str, province_code: Optional[str]
    ) -> List[PolicyChunk]:
        collection_name = self._collection_name(kb_scope, province_code)
        collection = self._get_or_create_collection(collection_name)
        query_embedding = self._embed([query])[0]
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        chunks: List[PolicyChunk] = []
        for doc, meta, dist in zip(docs, metas, distances):
            score = max(0.0, 1 - float(dist))
            chunks.append(PolicyChunk(text=doc, score=score, metadata=meta or {}))
        return chunks
