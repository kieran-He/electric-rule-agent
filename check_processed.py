from pathlib import Path
import json

for prov in ['CQ', 'FJ', 'JB', 'JL', 'JN', 'SN']:
    prov_dir = Path(f'data/processed/{prov}')
    json_files = list(prov_dir.glob('*.json'))
    print(f'{prov}: {len(json_files)} files')
    for jf in json_files[:1]:
        data = json.loads(jf.read_text(encoding='utf-8'))
        issuer = data.get('metadata', {}).get('issuer')
        doc_issuer = None
        if data.get('clauses'):
            doc_issuer = data.get('clauses', [])[0].get('doc_issuer')
        flags = data.get('processing_flags', [])
        print(f'  {jf.name}')
        print(f'    issuer: {issuer}')
        print(f'    doc_issuer: {doc_issuer}')
        print(f'    flags: {flags}')
        print()