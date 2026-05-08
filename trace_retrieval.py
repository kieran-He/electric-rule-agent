"""
Track each step of retrieval pipeline with detailed logging
"""
import sys
sys.path.insert(0, '.')

from app.config import settings
from app.core.repository import ChromaPolicyRepository
from app.langchain.query_rewriter import QueryRewriter, RewriteResult
from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker
from app.langchain.bm25_indexer import BM25Indexer
from app.langchain.llm import MiniMaxLLMWrapper
from dataprocess.bm25_builder import ProvinceBM25Indexer
import os

print('=' * 60)
print('Step 1: Query Rewrite')
print('=' * 60)

llm = MiniMaxLLMWrapper(
    api_key=os.getenv('LLM_API_KEY', ''),
    endpoint=os.getenv('LLM_ENDPOINT', 'https://api.minimaxi.com/anthropic'),
    model=os.getenv('LLM_MODEL', 'MiniMax-M2.7'),
)

rewriter = QueryRewriter(
    llm_wrapper=llm,
    enabled=True,
    always_rewrite=True,
)

original_query = '山东海南中长期电力市场交易对发电量的要求'
print(f'Original query: {original_query}')

rewrite_result = rewriter.rewrite(original_query)
print(f'Rewrite returned: {len(rewrite_result.queries)} queries')
print(f'Should split: {rewrite_result.should_split}')
print(f'Triggered: {rewrite_result.triggered}')

for i, qp in enumerate(rewrite_result.queries):
    print(f'  Query {i+1}: "{qp.query}"')
    print(f'    Provinces: {qp.province_codes}')

print()
print('=' * 60)
print('Step 2: Batch Embedding')
print('=' * 60)

repo = ChromaPolicyRepository(
    persist_directory=settings.chroma_path,
    embedding_model_name=settings.embedding_model,
)

queries = [qp.query for qp in rewrite_result.queries]
print(f'Queries to embed: {len(queries)}')

embeddings = repo.embed_queries_batch(queries)
print(f'Embeddings returned: {len(embeddings)}')
for q, emb in embeddings.items():
    print(f'  "{q[:30]}...": vector dim={len(emb)}')

print()
print('=' * 60)
print('Step 3: Vector Retrieval per Query')
print('=' * 60)

supported_provinces = ['HI', 'GS', 'SD', 'SN', 'AH', 'SX']
vector_results = {}

for i, qp in enumerate(rewrite_result.queries):
    province_codes = qp.province_codes if qp.province_codes else supported_provinces
    valid_codes = [c for c in province_codes if c in supported_provinces]
    
    print(f'Query {i+1}: "{qp.query[:40]}..."')
    print(f'  Province codes: {qp.province_codes} -> Valid: {valid_codes}')
    
    query_chunks = []
    for prov in valid_codes:
        chunks = repo.retrieve_with_embedding(
            embedding=embeddings[qp.query],
            top_k=settings.hybrid_vector_top_k,
            kb_scope='province',
            province_code=prov,
        )
        print(f'    Vector retrieval from {prov}: {len(chunks)} chunks')
        query_chunks.extend(chunks)
    
    vector_results[i] = query_chunks
    print(f'  Total vector chunks for Query {i+1}: {len(query_chunks)}')

print()
print('=' * 60)
print('Step 4: BM25 Retrieval per Query')
print('=' * 60)

bm25_results = {}

for i, qp in enumerate(rewrite_result.queries):
    province_codes = qp.province_codes if qp.province_codes else supported_provinces
    valid_codes = [c for c in province_codes if c in supported_provinces]
    
    print(f'Query {i+1}: "{qp.query[:40]}..."')
    
    query_chunks = []
    for prov in valid_codes:
        try:
            bm25 = ProvinceBM25Indexer(
                province_code=prov,
                processed_dir=f'data/processed/{prov}',
                cache_dir='data/cache',
            )
            bm25.build_index()
            results = bm25.search(qp.query, top_k=settings.hybrid_bm25_top_k)
            print(f'    BM25 retrieval from {prov}: {len(results)} chunks')
            for chunk_data, score in results:
                from app.core.repository import PolicyChunk
                chunk = PolicyChunk(
                    text=chunk_data['text'],
                    score=float(score),
                    metadata=chunk_data['metadata'],
                )
                query_chunks.append(chunk)
        except Exception as e:
            print(f'    BM25 retrieval from {prov}: FAILED ({e})')
    
    bm25_results[i] = query_chunks
    print(f'  Total BM25 chunks for Query {i+1}: {len(query_chunks)}')

print()
print('=' * 60)
print('Step 5: RRF Stage1 (per query Vector + BM25 fusion)')
print('=' * 60)

rrf_stage1_results = {}

for i in range(len(rewrite_result.queries)):
    vector_chunks = vector_results.get(i, [])
    bm25_chunks = bm25_results.get(i, [])
    
    print(f'Query {i+1}: Vector={len(vector_chunks)}, BM25={len(bm25_chunks)}')
    
    if len(vector_chunks) == 0 and len(bm25_chunks) == 0:
        print(f'  No chunks found! RRF Stage1 returns: 0 chunks')
        rrf_stage1_results[i] = []
        continue
    
    # Manual RRF Stage1
    from collections import defaultdict
    rrf_scores = defaultdict(float)
    chunk_map = {}
    
    k = settings.rrf_k
    top_k = settings.rrf_stage1_top_k
    
    for rank, chunk in enumerate(vector_chunks, start=1):
        chunk_hash = hash(chunk.text[:100])
        if chunk_hash not in chunk_map:
            chunk_map[chunk_hash] = chunk
        rrf_scores[chunk_hash] += 1.0 / (k + rank)
    
    for rank, chunk in enumerate(bm25_chunks, start=1):
        chunk_hash = hash(chunk.text[:100])
        if chunk_hash not in chunk_map:
            chunk_map[chunk_hash] = chunk
        rrf_scores[chunk_hash] += 1.0 / (k + rank)
    
    sorted_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)
    actual_top_k = min(top_k, len(sorted_hashes))
    
    result_chunks = []
    for h in sorted_hashes[:actual_top_k]:
        chunk = chunk_map[h]
        chunk.score = rrf_scores[h]
        result_chunks.append(chunk)
    
    rrf_stage1_results[i] = result_chunks
    print(f'  RRF Stage1 returns: {len(result_chunks)} chunks')

print()
print('=' * 60)
print('Step 6: RRF Stage2 (cross-query fusion)')
print('=' * 60)

per_query_results = [rrf_stage1_results[i] for i in range(len(rewrite_result.queries))]
print(f'Input: {len(per_query_results)} query results')

# Count chunks per query
for i, results in enumerate(per_query_results):
    print(f'  Query {i+1}: {len(results)} chunks')

if len(per_query_results) > 1:
    # Manual RRF Stage2
    from collections import defaultdict
    rrf_scores = defaultdict(float)
    chunk_map = {}
    
    k = settings.rrf_k
    top_k = settings.rrf_stage2_top_k
    
    for query_idx, query_results in enumerate(per_query_results):
        print(f'Processing Query {query_idx+1} with {len(query_results)} chunks')
        for rank, chunk in enumerate(query_results, start=1):
            chunk_hash = hash(chunk.text[:100])
            if chunk_hash not in chunk_map:
                chunk_map[chunk_hash] = chunk
            rrf_scores[chunk_hash] += 1.0 / (k + rank)
    
    sorted_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)
    actual_top_k = min(top_k, len(sorted_hashes))
    
    all_candidates = []
    for h in sorted_hashes[:actual_top_k]:
        chunk = chunk_map[h]
        chunk.score = rrf_scores[h]
        all_candidates.append(chunk)
    
    print(f'RRF Stage2 returns: {len(all_candidates)} candidates')
elif len(per_query_results) == 1:
    all_candidates = per_query_results[0]
    print(f'Only 1 query, RRF Stage2 skipped. Using Stage1 result: {len(all_candidates)} candidates')
else:
    all_candidates = []
    print(f'No query results! RRF Stage2 returns: 0 candidates')

print()
print('=' * 60)
print('Step 7: Rerank')
print('=' * 60)

reranker = BGEReranker(
    model_name=settings.reranker_model,
    max_length=settings.reranker_max_length,
)

rerank_query = rewrite_result.queries[0].query if rewrite_result.queries else original_query
print(f'Rerank query: "{rerank_query[:40]}..."')
print(f'Candidates to rerank: {len(all_candidates)}')

if len(all_candidates) > settings.hybrid_final_top_k:
    final_chunks = reranker.rerank(rerank_query, all_candidates, top_k=settings.hybrid_final_top_k)
else:
    final_chunks = all_candidates[:settings.hybrid_final_top_k]

print(f'Final chunks after rerank: {len(final_chunks)}')

for i, chunk in enumerate(final_chunks[:5]):
    has_gen = '发电量' in chunk.text
    print(f'  Chunk {i+1}: Score={chunk.score:.3f}, 发电量={has_gen}')
    print(f'    Source: {chunk.metadata.get("doc_name", "N/A")[:40]}...')

print()
print('=' * 60)
print('Summary')
print('=' * 60)

print(f'Original query: 1')
print(f'Rewritten queries: {len(rewrite_result.queries)}')
for i in range(len(rewrite_result.queries)):
    print(f'  Query {i+1}: Vector={len(vector_results.get(i, []))}, BM25={len(bm25_results.get(i, []))}, RRF1={len(rrf_stage1_results.get(i, []))}')
print(f'RRF Stage2 candidates: {len(all_candidates)}')
print(f'Final chunks: {len(final_chunks)}')

print()
print('Special case analysis:')
for i in range(len(rewrite_result.queries)):
    if len(rrf_stage1_results.get(i, [])) == 0:
        print(f'Query {i+1} had NO chunks in Stage1')
        print(f'  It contributed 0 to Stage2 RRF scores')
        print(f'  Other queries\' chunks will dominate Stage2')