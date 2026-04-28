"""
BM25 Indexer for Chinese Text Retrieval

Uses jieba tokenizer with custom dictionary for power policy domain.
Configurable k1 and b parameters for optimal retrieval.
Supports persistent cache to avoid rebuilding index on startup.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import List, Tuple

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    jieba = None

from app.core.repository import PolicyChunk
from app.langchain.bm25_params import BM25WithParams


class BM25Indexer:
    """
    BM25 Indexer with jieba tokenization for Chinese text.
    
    Features:
    - Custom dictionary for power policy domain terms
    - Stopword filtering
    - Configurable k1 and b parameters for optimal retrieval
    - Persistent cache support to avoid rebuilding on startup
    """
    
    def __init__(
        self,
        corpus_path: str = "data/processed",
        dict_path: str = "data/dict/power_policy.txt",
        stopwords_path: str = "data/dict/stopwords.txt",
        k1: float = 1.5,
        b: float = 0.6,
        cache_path: str = "data/cache/bm25_index.pkl",
    ):
        self.corpus_path = Path(corpus_path)
        self.dict_path = Path(dict_path)
        self.stopwords_path = Path(stopwords_path)
        self.cache_path = Path(cache_path)
        self.k1 = k1
        self.b = b
        
        self.bm25: BM25WithParams | BM25Okapi | None = None
        self.documents: List[str] = []
        self.metadatas: List[dict] = []
        self.tokenized_corpus: List[List[str]] = []
        
        self._setup_tokenizer()
    
    def _setup_tokenizer(self) -> None:
        """Setup jieba tokenizer with custom dictionary."""
        if not JIEBA_AVAILABLE:
            print("Warning: jieba not installed, using character-level tokenization")
            return
        
        if self.dict_path.exists():
            jieba.load_userdict(str(self.dict_path))
            print(f"Loaded custom dictionary: {self.dict_path}")
        
        self.stopwords: set[str] = set()
        if self.stopwords_path.exists():
            with open(self.stopwords_path, encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
            print(f"Loaded stopwords: {len(self.stopwords)} words")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Chinese text.
        
        Uses jieba if available, otherwise character-level tokenization.
        """
        if JIEBA_AVAILABLE:
            tokens = jieba.lcut(text)
        else:
            tokens = list(text)
        
        return [t for t in tokens if t not in self.stopwords and len(t.strip()) > 1]
    
    def build_index(self) -> int:
        """
        Build or load BM25 index from cache.
        
        Returns:
            Number of documents indexed
        """
        corpus_hash = self._compute_corpus_hash()
        
        if self._load_cache(corpus_hash):
            print(f"BM25 index loaded from cache: {len(self.documents)} documents (k1={self.k1}, b={self.b})")
            return len(self.documents)
        
        total_docs = self._build_from_corpus()
        
        if total_docs > 0:
            self._save_cache(corpus_hash)
            print(f"BM25 index built and cached: {total_docs} documents (k1={self.k1}, b={self.b})")
        
        return total_docs
    
    def _compute_corpus_hash(self) -> str:
        """
        Compute hash of corpus files to detect changes.
        
        Returns:
            MD5 hash of all corpus files combined with dict files
        """
        hash_md5 = hashlib.md5()
        
        json_files = sorted(self.corpus_path.glob("*.json"))
        json_files = [f for f in json_files if not f.name.startswith("_")]
        
        for json_file in json_files:
            try:
                hash_md5.update(json_file.read_bytes())
            except Exception:
                pass
        
        if self.dict_path.exists():
            hash_md5.update(self.dict_path.read_bytes())
        
        if self.stopwords_path.exists():
            hash_md5.update(self.stopwords_path.read_bytes())
        
        hash_md5.update(f"k1={self.k1},b={self.b}".encode())
        
        return hash_md5.hexdigest()
    
    def _load_cache(self, expected_hash: str) -> bool:
        """
        Load BM25 index from cache if valid.
        
        Args:
            expected_hash: Expected corpus hash
            
        Returns:
            True if cache loaded successfully, False otherwise
        """
        if not self.cache_path.exists():
            return False
        
        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if cache_data.get('hash') != expected_hash:
                print(f"Cache hash mismatch, rebuilding index")
                return False
            
            self.documents = cache_data['documents']
            self.metadatas = cache_data['metadatas']
            self.tokenized_corpus = cache_data['tokenized_corpus']
            self.bm25 = cache_data['bm25']
            
            return True
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return False
    
    def _save_cache(self, corpus_hash: str) -> None:
        """
        Save BM25 index to cache.
        
        Args:
            corpus_hash: Current corpus hash
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            'hash': corpus_hash,
            'k1': self.k1,
            'b': self.b,
            'documents': self.documents,
            'metadatas': self.metadatas,
            'tokenized_corpus': self.tokenized_corpus,
            'bm25': self.bm25,
        }
        
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"BM25 index cached to: {self.cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _build_from_corpus(self) -> int:
        """
        Build BM25 index from corpus files.
        
        Returns:
            Number of documents indexed
        """
        json_files = list(self.corpus_path.glob("*.json"))
        json_files = [f for f in json_files if not f.name.startswith("_")]
        
        total_docs = 0
        
        for json_file in json_files:
            try:
                with open(json_file, encoding='utf-8') as f:
                    data = json.load(f)
                
                clauses = data.get("clauses", [])
                for clause in clauses:
                    text = clause.get("clause_text", "")
                    if not text:
                        continue
                    
                    self.documents.append(text)
                    self.metadatas.append({
                        "doc_name": clause.get("doc_name", ""),
                        "source_name": clause.get("doc_name", ""),
                        "title_path": clause.get("title_path", ""),
                        "article_no": clause.get("article_no", ""),
                        "policy_level": clause.get("policy_level", "formal"),
                        "file_hash": json_file.stem,
                        "page_start": clause.get("page_start"),
                        "page_end": clause.get("page_end"),
                    })
                    total_docs += 1
                    
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
        
        if not self.documents:
            print("Warning: No documents found for BM25 indexing")
            return 0
        
        self.tokenized_corpus = [self.tokenize(doc) for doc in self.documents]
        
        self.bm25 = BM25WithParams(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )
        
        return len(self.documents)
    
    def search(self, query: str, top_k: int = 12) -> List[Tuple[PolicyChunk, float]]:
        """
        Search documents using BM25.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (PolicyChunk, score) tuples
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        
        results: List[Tuple[PolicyChunk, float]] = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] <= 0:
                continue
            
            chunk = PolicyChunk(
                text=self.documents[idx],
                score=float(scores[idx]),
                metadata=self.metadatas[idx],
            )
            results.append((chunk, scores[idx]))
        
        return results
    
    def is_available(self) -> bool:
        """Check if BM25 indexer is ready."""
        return self.bm25 is not None
    
    def get_stats(self) -> dict:
        """Get statistics about the index."""
        stats = {
            "documents": len(self.documents),
            "available": self.is_available(),
            "jieba": JIEBA_AVAILABLE,
            "bm25": BM25_AVAILABLE,
            "k1": self.k1,
            "b": self.b,
            "cache_path": str(self.cache_path),
            "cache_exists": self.cache_path.exists(),
        }
        
        if self.bm25 is not None and isinstance(self.bm25, BM25WithParams):
            params = self.bm25.get_params()
            stats["avgdl"] = params.get("avgdl", 0)
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the cache file."""
        if self.cache_path.exists():
            self.cache_path.unlink()
            print(f"BM25 cache cleared: {self.cache_path}")