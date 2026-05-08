"""
Direct RRF vs Simple Concatenation Test

Test RRF methods directly without reranker to see true fusion quality.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.core.repository import ChromaPolicyRepository, PolicyChunk
from app.langchain.bm25_indexer import BM25Indexer
from dataprocess.bm25_builder import ProvinceBM25Indexer


def test_rrf_stage1_direct():
    """Test Stage1 RRF directly without HybridRetriever wrapper."""
    
    print("Initializing repositories...")
    repo = ChromaPolicyRepository(
        persist_directory=settings.chroma_path,
        embedding_model_name=settings.embedding_model,
    )
    
    bm25_global = BM25Indexer(k1=settings.bm25_k1, b=settings.bm25_b)
    bm25_global.build_index()
    
    bm25_province = ProvinceBM25Indexer(
        province_code="SN",
        processed_dir="data/processed/SN",
        cache_dir="data/cache",
        k1=1.5,
        b=0.6,
    )
    bm25_province.build_index()
    
    query = "陕西中长期交易流程"
    province = "SN"
    vector_top_k = 12
    bm25_top_k = 12
    
    print(f"\n=== Direct Stage1 RRF Test ===")
    print(f"Query: {query}")
    print(f"Province: {province}")
    print(f"Vector top_k: {vector_top_k}, BM25 top_k: {bm25_top_k}")
    
    vector_chunks = repo.retrieve(
        query=query,
        top_k=vector_top_k,
        kb_scope="province",
        province_code=province,
    )
    
    bm25_results = bm25_province.search(query, top_k=bm25_top_k)
    bm25_chunks = []
    for chunk_data, score in bm25_results:
        chunk = PolicyChunk(
            text=chunk_data["text"],
            score=float(score),
            metadata=chunk_data["metadata"],
        )
        bm25_chunks.append(chunk)
    
    print(f"\nVector results ({len(vector_chunks)} chunks):")
    for i, c in enumerate(vector_chunks[:5], 1):
        source = c.metadata.get("source_name", "unknown")[:40]
        print(f"  {i}. score={c.score:.4f} | {source}")
    
    print(f"\nBM25 results ({len(bm25_chunks)} chunks):")
    for i, c in enumerate(bm25_chunks[:5], 1):
        source = c.metadata.get("source_name", "unknown")[:40]
        print(f"  {i}. score={c.score:.4f} | {source}")
    
    rrf_scores = {}
    chunk_map = {}
    k = 60
    top_k = 15
    
    for rank, chunk in enumerate(vector_chunks, start=1):
        chunk_hash = hash(chunk.text[:100])
        if chunk_hash not in rrf_scores:
            rrf_scores[chunk_hash] = 0.0
            chunk_map[chunk_hash] = chunk
        rrf_scores[chunk_hash] += 1.0 / (k + rank)
    
    for rank, chunk in enumerate(bm25_chunks, start=1):
        chunk_hash = hash(chunk.text[:100])
        if chunk_hash not in rrf_scores:
            rrf_scores[chunk_hash] = 0.0
            chunk_map[chunk_hash] = chunk
        rrf_scores[chunk_hash] += 1.0 / (k + rank)
    
    sorted_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)
    
    rrf_chunks = []
    actual_top_k = min(top_k, len(sorted_hashes))
    for h in sorted_hashes[:actual_top_k]:
        chunk = chunk_map[h]
        chunk.score = rrf_scores[h]
        rrf_chunks.append(chunk)
    
    print(f"\nRRF fusion results ({len(rrf_chunks)} chunks):")
    for i, c in enumerate(rrf_chunks[:10], 1):
        source = c.metadata.get("source_name", "unknown")[:40]
        print(f"  {i}. score={c.score:.4f} | {source}")
    
    seen_hashes = set()
    simple_chunks = []
    for chunk in vector_chunks + bm25_chunks:
        chunk_hash = hash(chunk.text[:100])
        if chunk_hash not in seen_hashes:
            seen_hashes.add(chunk_hash)
            simple_chunks.append(chunk)
    
    print(f"\nSimple concatenation results ({len(simple_chunks)} chunks):")
    for i, c in enumerate(simple_chunks[:10], 1):
        source = c.metadata.get("source_name", "unknown")[:40]
        print(f"  {i}. score={c.score:.4f} | {source}")
    
    hashes_rrf = [hash(c.text[:100]) for c in rrf_chunks[:10]]
    hashes_simple = [hash(c.text[:100]) for c in simple_chunks[:10]]
    
    overlap = len(set(hashes_rrf) & set(hashes_simple))
    print(f"\nOverlap ratio top10: {overlap}/10 = {overlap/10:.1%}")
    
    order_diff_count = 0
    for i in range(min(10, len(hashes_rrf), len(hashes_simple))):
        if hashes_rrf[i] != hashes_simple[i]:
            order_diff_count += 1
    print(f"Order differences in top10: {order_diff_count}")
    
    result = {
        "query": query,
        "vector_count": len(vector_chunks),
        "bm25_count": len(bm25_chunks),
        "rrf_count": len(rrf_chunks),
        "simple_count": len(simple_chunks),
        "overlap_top10": overlap,
        "order_diff_count": order_diff_count,
        "rrf_scores": [round(c.score, 4) for c in rrf_chunks[:10]],
        "simple_scores": [round(c.score, 4) for c in simple_chunks[:10]],
        "rrf_sources": [c.metadata.get("source_name", "unknown")[:40] for c in rrf_chunks[:10]],
        "simple_sources": [c.metadata.get("source_name", "unknown")[:40] for c in simple_chunks[:10]],
    }
    
    return result


def main():
    result = test_rrf_stage1_direct()
    
    output_path = Path("data/rrf_raw_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()