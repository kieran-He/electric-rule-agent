#!/usr/bin/env python3
"""
Prepare 20-question test set from benchmark.json

Distribution: clause_qa(8), flow_qa(4), compare_qa(3), settlement_qa(3), rejection(2)
"""
import json
from pathlib import Path

def prepare_test_set():
    """从benchmark.json抽取20条测试集"""
    benchmark_path = Path("evaluation/benchmark.json")
    
    with open(benchmark_path, encoding='utf-8') as f:
        data = json.load(f)
    
    questions = data.get("questions", [])
    
    # 按类别分组
    categories = {}
    for q in questions:
        cat = q.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(q)
    
    print(f"Categories found: {list(categories.keys())}")
    for cat, qs in categories.items():
        print(f"  {cat}: {len(qs)} questions")
    
    # 分层抽样: clause_qa(8), flow_qa(4), compare_qa(3), settlement_qa(3), rejection(2)
    test_set = []
    test_set.extend(categories.get("clause_qa", [])[:8])
    test_set.extend(categories.get("flow_qa", [])[:4])
    test_set.extend(categories.get("compare_qa", [])[:3])
    test_set.extend(categories.get("settlement_qa", [])[:3])
    test_set.extend(categories.get("rejection", [])[:2])
    
    print(f"\nSelected: {len(test_set)} questions")
    
    # 统计分布
    dist = {}
    for q in test_set:
        cat = q.get("category", "unknown")
        dist[cat] = dist.get(cat, 0) + 1
    print(f"Distribution: {dist}")
    
    # 保存
    output = {
        "version": "test_v2_bm25",
        "generated_at": "2026-04-23",
        "total_count": len(test_set),
        "distribution": dist,
        "questions": test_set
    }
    
    output_path = Path("evaluation/benchmark_test.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to: {output_path}")
    
    # 显示问题列表
    print("\nQuestion list:")
    for i, q in enumerate(test_set):
        print(f"  [{i+1}] {q.get('question_id')}: {q.get('question')[:60]}... ({q.get('category')})")

if __name__ == "__main__":
    prepare_test_set()