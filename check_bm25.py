from dataprocess.bm25_builder import ProvinceBM25Indexer
from app.langchain.bm25_indexer import BM25Indexer

print('=== Global BM25Indexer ===')
global_bm25 = BM25Indexer(
    corpus_path='data/processed',
    k1=1.5,
    b=0.6,
    cache_path='data/cache/bm25_index.pkl'
)
global_count = global_bm25.build_index()
print(f'Global BM25: {global_count} documents')

results = global_bm25.search('山东中长期电力市场交易对发电量的要求', top_k=5)
print(f'Global search results: {len(results)}')
for i, (chunk, score) in enumerate(results[:3]):
    print(f'{i+1}. Score={score:.3f}, Province={chunk.metadata.get("province_code")}, Text={chunk.text[:60]}...')

print()
print('=== ProvinceBM25Indexer (SD) ===')
try:
    sd_bm25 = ProvinceBM25Indexer(
        province_code='SD',
        processed_dir='data/processed/SD',
        cache_dir='data/cache',
        k1=1.5,
        b=0.6
    )
    sd_count = sd_bm25.build_index()
    print(f'SD BM25: {sd_count} documents')
    
    results = sd_bm25.search('山东省中长期电力市场交易规则对发电量的要求', top_k=5)
    print(f'SD search results: {len(results)}')
    for i, (chunk_data, score) in enumerate(results[:3]):
        print(f'{i+1}. Score={score:.3f}, Text={chunk_data["text"][:60]}...')
except Exception as e:
    print(f'SD BM25 error: {e}')