#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from pathlib import Path
import json

processed_dir = Path("data/processed")
json_files = [f for f in processed_dir.glob("*.json") if not f.name.startswith("_")]

docs_info = []
for json_file in json_files:
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    doc_name = meta.get("doc_name", "")
    doc_type = meta.get("doc_type", "")
    clauses_count = len(data.get("clauses", []))
    docs_info.append({
        "file": json_file.stem,
        "doc_name": doc_name,
        "doc_type": doc_type,
        "clauses": clauses_count
    })

print("Documents in processed directory:")
print("-" * 80)
for d in docs_info:
    print(f"File: {d['file'][:20]}...")
    print(f"  Name: {d['doc_name'][:60]}...")
    print(f"  Type: {d['doc_type']}")
    print(f"  Clauses: {d['clauses']}")
    print()