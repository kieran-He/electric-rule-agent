import hashlib
import math
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


class DeterministicEmbedder:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dimension
            for token in text.split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], byteorder="little", signed=False) % self.dimension
                vec[idx] += 1.0
            if not any(vec) and text:
                for ch in text:
                    digest = hashlib.md5(ch.encode("utf-8")).digest()
                    idx = int.from_bytes(digest[:2], byteorder="little", signed=False) % self.dimension
                    vec[idx] += 1.0
            if normalize_embeddings:
                norm = math.sqrt(sum(v * v for v in vec))
                if norm > 0:
                    vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


class ChromaPolicyRepository:
    def __init__(self, persist_directory: str, embedding_model_name: str):
        self._ready = False
        self._client = None
        self._embedder = None
        self._persist_directory = Path(persist_directory)
        self._embedding_model_name = embedding_model_name
        self._embedder_name = embedding_model_name

        try:
            import chromadb

            self._persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_directory))
        except Exception as exc:
            self._init_error = str(exc)
            return

        if self._embedding_model_name.lower() in {
            "deterministic",
            "deterministic-fallback",
            "fallback",
        }:
            self._embedder = DeterministicEmbedder()
            self._embedder_name = "deterministic-fallback"
            self._ready = True
            return

        try:
            from sentence_transformers import SentenceTransformer
            from app.core.embedding_cache import embedding_cache

            self._embedder = embedding_cache.preload(self._embedding_model_name)
            if self._embedder is None:
                self._embedder = SentenceTransformer(self._embedding_model_name)
        except Exception:
            self._embedder = DeterministicEmbedder()
            self._embedder_name = "deterministic-fallback"
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def init_error(self) -> Optional[str]:
        return getattr(self, "_init_error", None)

    @property
    def embedder_name(self) -> str:
        return self._embedder_name

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
        if hasattr(vectors, "tolist"):
            return vectors.tolist()
        return vectors

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