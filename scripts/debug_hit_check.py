import re
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def normalize_doc_name(name):
    if not name:
        return ''
    name = re.sub(r'附件\d*[：:]', '', name)
    name = re.sub(r'转载丨', '', name)
    name = re.sub(r'[《<>》]', '', name)
    name = re.sub(r'[（(][^）)]*[）)]', '', name)
    name = re.sub(r'\d{4}年\d*月?', '', name)
    name = re.sub(r'〔\d+〕', '', name)
    name = re.sub(r'（修订版|V\d+|征求意见稿|连续试运行|试运行）', '', name)
    name = re.sub(r'\s+', '', name)
    return name.strip()

def extract_keywords(doc_name):
    if not doc_name:
        return []
    
    keywords = []
    patterns = [
        r'陕西|陕西',
        r'电力',
        r'中长期|中长期',
        r'现货|现货',
        r'分时段|分时段',
        r'零售|零售',
        r'交易|交易',
        r'结算|结算',
        r'实施细则|实施细则',
        r'交易细则|交易细则',
        r'调频|调频',
        r'辅助服务|辅助',
        r'新型储能|储能',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, doc_name)
        if match:
            keywords.append(match.group())
    
    return keywords if len(keywords) >= 3 else []

with open('evaluation/reports_hybrid/retrieval_metrics_shaaxi.json', encoding='utf-8') as f:
    report = json.load(f)

for detail in report['details']:
    if detail['question_id'] in ['q005', 'q007', 'q008', 'q010']:
        qid = detail['question_id']
        expected = detail['expected_docs']
        retrieved = detail['hybrid_docs']
        print(f'=== {qid} ===')
        print(f'Expected: {expected}')
        print(f'Retrieved Top1: {retrieved[0] if retrieved else "empty"}')
        
        if expected and retrieved:
            exp_norm = normalize_doc_name(expected[0])
            ret_norm = normalize_doc_name(retrieved[0])
            print(f'Expected normalized: {exp_norm}')
            print(f'Retrieved normalized: {ret_norm}')
            
            keywords = extract_keywords(exp_norm)
            print(f'Keywords extracted: {keywords}')
            if keywords:
                keyword_hit = all(kw in ret_norm for kw in keywords)
                print(f'Keyword hit: {keyword_hit}')
                for kw in keywords:
                    print(f'  "{kw}" in ret_norm: {kw in ret_norm}')
        print()