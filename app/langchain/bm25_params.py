"""
BM25 with Adjustable Parameters

Custom BM25 implementation allowing k1 and b parameter tuning for optimal retrieval.
Default BM25Okapi doesn't allow parameter adjustment.

BM25 scoring formula:
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))

Where:
- k1: Term frequency saturation parameter (default 1.2, recommended 1.5 for policy docs)
- b: Document length normalization (default 0.75, recommended 0.6 for policy docs)
"""
from __future__ import annotations

import math
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class BM25WithParams:
    """
    BM25 implementation with adjustable k1 and b parameters.
    
    Optimized for Chinese policy document retrieval.
    
    Args:
        corpus: List of tokenized documents (each doc is list of tokens)
        k1: Term frequency saturation parameter (default 1.5)
        b: Document length normalization (default 0.6)
        epsilon: Floor value for IDF (default 0.25)
    """
    
    def __init__(
        self,
        corpus: List[List[str]],
        k1: float = 1.5,
        b: float = 0.6,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs: List[dict] = []
        self.idf: dict = {}
        self.doc_len: List[int] = []
        
        if corpus:
            self._initialize()
    
    def _initialize(self) -> None:
        """Initialize BM25 parameters and compute IDF values."""
        nd = {}
        num_doc = 0
        
        for document in self.corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)
            
            frequencies: dict = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)
            
            for word in frequencies.keys():
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1
        
        self.avgdl = num_doc / self.corpus_size if self.corpus_size > 0 else 0
        
        idf_sum = 0
        idf_len = 0
        negative_idfs = []
        
        for word, freq in nd.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf
            idf_sum += idf
            idf_len += 1
            
            if idf < 0:
                negative_idfs.append(word)
        
        if negative_idfs:
            avg_idf = idf_sum / idf_len if idf_len > 0 else 0
            eps = self.epsilon * avg_idf
            for word in negative_idfs:
                self.idf[word] = eps
        
        logger.debug(f"BM25 initialized: corpus_size={self.corpus_size}, avgdl={self.avgdl:.2f}, k1={self.k1}, b={self.b}")
    
    def get_scores(self, query: List[str]) -> List[float]:
        """
        Compute BM25 scores for a query against all documents.
        
        Args:
            query: Tokenized query (list of tokens)
            
        Returns:
            List of BM25 scores for each document
        """
        scores = []
        
        for doc_idx in range(self.corpus_size):
            score = 0.0
            doc_freqs = self.doc_freqs[doc_idx]
            
            for word in query:
                if word not in self.idf:
                    continue
                
                idf = self.idf[word]
                
                if word in doc_freqs:
                    freq = doc_freqs[word]
                    doc_len = self.doc_len[doc_idx]
                    
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * numerator / denominator
            
            scores.append(score)
        
        return scores
    
    def get_top_n(
        self,
        query: List[str],
        n: int = 10,
    ) -> List[int]:
        """
        Get top-n document indices for a query.
        
        Args:
            query: Tokenized query
            n: Number of results
            
        Returns:
            List of document indices sorted by score
        """
        scores = self.get_scores(query)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        return ranked_indices[:n]
    
    def get_params(self) -> dict:
        """Get current BM25 parameters."""
        return {
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
        }


def create_bm25(
    corpus: List[List[str]],
    k1: float = 1.5,
    b: float = 0.6,
) -> BM25WithParams:
    """
    Factory function to create BM25 instance with custom parameters.
    
    Args:
        corpus: List of tokenized documents
        k1: Term frequency saturation parameter
        b: Document length normalization
        
    Returns:
        BM25WithParams instance
    """
    return BM25WithParams(corpus, k1=k1, b=b)