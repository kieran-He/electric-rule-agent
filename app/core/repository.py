import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class PolicyChunk:
    text: str
    score: float
    metadata: Dict[str, str]


class RepositoryError(RuntimeError):
    pass


class DeterministicEmbedder:
    def __init__(self, dimension: int = 512):
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
        self._query_embed_cache: Dict[str, List[float]] = {}
        self._cache_max_size = 500
        self._cache_hits = 0
        self._cache_misses = 0

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
    
    def embed_queries_batch(self, queries: List[str]) -> Dict[str, List[float]]:
        """
        Batch embed multiple queries with caching.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dict mapping query text to embedding vector
        """
        if not self._ready:
            raise RepositoryError(self.init_error or "repository not ready")
        
        cached_embeddings = {}
        uncached_queries = []
        
        for query in queries:
            if query in self._query_embed_cache:
                cached_embeddings[query] = self._query_embed_cache[query]
                self._cache_hits += 1
            else:
                uncached_queries.append(query)
                self._cache_misses += 1
        
        if uncached_queries:
            hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) * 100
            logger.info(f"Batch embedding {len(uncached_queries)} queries (cached: {len(cached_embeddings)}, hit_rate: {hit_rate:.1f}%)")
            
            vectors = self._embedder.encode(uncached_queries, normalize_embeddings=True)
            
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()
            
            for i, query in enumerate(uncached_queries):
                embedding = vectors[i]
                cached_embeddings[query] = embedding
                self._query_embed_cache[query] = embedding
                
                if len(self._query_embed_cache) > self._cache_max_size:
                    self._trim_cache()
        
        if self._cache_hits > 0 or self._cache_misses > 0:
            from app.core.metrics import metrics_store
            metrics_store.record_cache_stats(
                cache_type="query_embed",
                hits=self._cache_hits,
                misses=self._cache_misses,
                size=len(self._query_embed_cache)
            )
        
        return cached_embeddings
    
    def retrieve_with_embedding(
        self,
        embedding: List[float],
        top_k: int,
        kb_scope: str,
        province_code: Optional[str]
    ) -> List[PolicyChunk]:
        """
        Retrieve using pre-computed embedding.
        
        Args:
            embedding: Pre-computed query embedding vector
            top_k: Number of results
            kb_scope: Knowledge base scope
            province_code: Province code
            
        Returns:
            List of PolicyChunk
        """
        collection_name = self._collection_name(kb_scope, province_code)
        collection = self._get_or_create_collection(collection_name)
        
        result = collection.query(
            query_embeddings=[embedding],
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
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests * 100 if total_requests > 0 else 0
        
        return {
            "cache_type": "query_embed",
            "cache_size": len(self._query_embed_cache),
            "max_size": self._cache_max_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
        }
    
    def _trim_cache(self):
        """Trim cache by removing oldest 50% entries."""
        keys_to_remove = list(self._query_embed_cache.keys())[:self._cache_max_size // 2]
        for k in keys_to_remove:
            del self._query_embed_cache[k]
        logger.debug(f"Trimmed embed cache: removed {len(keys_to_remove)} entries")