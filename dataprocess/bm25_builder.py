from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from dataprocess.config import DocProcSettings, settings
from dataprocess.schemas import ProcessedDocument


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


class BM25BuildError(RuntimeError):
    pass


class BM25WithParams:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.6):
        if BM25_AVAILABLE:
            self._bm25 = BM25Okapi(corpus)
            self._bm25.k1 = k1
            self._bm25.b = b
        else:
            self._corpus = corpus
            self._k1 = k1
            self._b = b
        self._corpus = corpus
        self._k1 = k1
        self._b = b
    
    def get_scores(self, query: list[str]) -> list[float]:
        if BM25_AVAILABLE and self._bm25:
            return self._bm25.get_scores(query)
        return [0.0] * len(self._corpus)
    
    def get_params(self) -> dict[str, Any]:
        return {"k1": self._k1, "b": self._b, "corpus_size": len(self._corpus)}


class ProvinceBM25Indexer:
    def __init__(
        self,
        province_code: str,
        processed_dir: str | Path,
        cache_dir: str | Path | None = None,
        dict_path: str | Path | None = None,
        stopwords_path: str | Path | None = None,
        k1: float = 1.5,
        b: float = 0.6,
        config: DocProcSettings | None = None,
    ):
        cfg = config or settings
        self.province_code = province_code.upper()
        self.processed_dir = Path(processed_dir)
        self.cache_dir = Path(cache_dir or cfg.cache_path)
        self.dict_path = Path(dict_path or "data/dict/power_policy.txt")
        self.stopwords_path = Path(stopwords_path or "data/dict/stopwords.txt")
        self.k1 = k1
        self.b = b
        
        self.cache_path = self.cache_dir / f"bm25_{self.province_code.lower()}.pkl"
        
        self.bm25: BM25WithParams | None = None
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []
        self.stopwords: set[str] = set()
        
        self._setup_tokenizer()
    
    def _setup_tokenizer(self) -> None:
        if not JIEBA_AVAILABLE:
            return
        
        if self.dict_path.exists():
            jieba.load_userdict(str(self.dict_path))
        
        if self.stopwords_path.exists():
            with open(self.stopwords_path, encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
    
    def tokenize(self, text: str) -> list[str]:
        if JIEBA_AVAILABLE:
            tokens = jieba.lcut(text)
        else:
            tokens = list(text)
        
        return [t for t in tokens if t not in self.stopwords and len(t.strip()) > 1]
    
    def build_index(self) -> int:
        corpus_hash = self._compute_corpus_hash()
        
        if self._load_cache(corpus_hash):
            return len(self.documents)
        
        total_docs = self._build_from_corpus()
        
        if total_docs > 0:
            self._save_cache(corpus_hash)
        
        return total_docs
    
    def _compute_corpus_hash(self) -> str:
        hash_md5 = hashlib.md5()
        
        json_files = sorted(self.processed_dir.glob("*.json"))
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
        if not self.cache_path.exists():
            return False
        
        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if cache_data.get('hash') != expected_hash:
                return False
            
            self.documents = cache_data['documents']
            self.metadatas = cache_data['metadatas']
            self.tokenized_corpus = cache_data['tokenized_corpus']
            self.bm25 = cache_data['bm25']
            
            return True
        except Exception:
            return False
    
    def _save_cache(self, corpus_hash: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
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
        except Exception:
            pass
    
    def _build_from_corpus(self) -> int:
        json_files = list(self.processed_dir.glob("*.json"))
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
                        "policy_level": clause.get("doc_status", "formal"),
                        "file_hash": json_file.stem,
                        "page_start": clause.get("page_start"),
                        "page_end": clause.get("page_end"),
                    })
                    total_docs += 1
                    
            except Exception:
                pass
        
        if not self.documents:
            return 0
        
        self.tokenized_corpus = [self.tokenize(doc) for doc in self.documents]
        
        self.bm25 = BM25WithParams(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )
        
        return len(self.documents)
    
    def search(self, query: str, top_k: int = 12) -> list[tuple]:
        if self.bm25 is None:
            raise BM25BuildError("BM25 index not built. Call build_index() first.")
        
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        
        results: list[tuple] = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] <= 0:
                continue
            
            chunk_data = {
                "text": self.documents[idx],
                "score": float(scores[idx]),
                "metadata": self.metadatas[idx],
            }
            results.append((chunk_data, scores[idx]))
        
        return results
    
    def get_stats(self) -> dict[str, Any]:
        return {
            "province_code": self.province_code,
            "documents": len(self.documents),
            "bm25_available": BM25_AVAILABLE,
            "jieba_available": JIEBA_AVAILABLE,
            "k1": self.k1,
            "b": self.b,
            "cache_path": str(self.cache_path),
            "cache_exists": self.cache_path.exists(),
        }
    
    def clear_cache(self) -> None:
        if self.cache_path.exists():
            self.cache_path.unlink()