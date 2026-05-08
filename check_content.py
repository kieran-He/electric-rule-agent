import chromadb
from app.core.repository import ChromaPolicyRepository

client = chromadb.PersistentClient(path='./data/chroma')

print('=== Vector Search for query ===')
repo = ChromaPolicyRepository(
    persist_directory='./data/chroma',
    embedding_model_name='BAAI/bge-small-zh-v1.5'
)

chunks = repo.retrieve(
    query='山东省中长期电力市场交易规则对发电量的要求',
    top_k=8,
    kb_scope='province',
    province_code='SD'
)

print(f'Vector search returned {len(chunks)} chunks')
for i, chunk in enumerate(chunks[:8]):
    has_gen = '发电量' in chunk.text
    has_long = '中长期' in chunk.text
    print(f'{i+1}. Score={chunk.score:.3f}, 发电量={has_gen}, 中长期={has_long}')
    print(f'   Source: {chunk.metadata.get("doc_name", "N/A")}')
    print(f'   Title: {chunk.metadata.get("title_path", "N/A")}')
    print(f'   Text: {chunk.text[:80]}...')
    print()

print('=== Check all documents for keywords ===')
sd_collection = client.get_collection('kb_sd')
all_docs = sd_collection.get(limit=1225, include=['documents'])

gen_count = 0
for doc in all_docs['documents']:
    if '发电量' in doc:
        gen_count += 1

print(f'Documents containing "发电量": {gen_count}/{len(all_docs["documents"])}')

long_count = 0
for doc in all_docs['documents']:
    if '中长期' in doc:
        long_count += 1

print(f'Documents containing "中长期": {long_count}/{len(all_docs["documents"])}')