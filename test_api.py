import requests
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 测试 agent/chat 接口
url = "http://localhost:8000/agent/chat"
payload = {
    "query": "电力市场化交易流程",
    "session_id": "test_chunk_001",
    "province_codes": ["SN"],
    "show_chunks": True
}

print("【测试Agent接口chunk格式化】")
print(f"请求: {payload}")

try:
    resp = requests.post(url, json=payload, timeout=180)
    data = resp.json()
    
    print(f"\n状态码: {resp.status_code}")
    print(f"工具调用: {data.get('tool_calls', [])}")
    
    answer = data.get('answer', '')
    
    # 验证chunk格式化
    has_ref_section = '**相关参考**' in answer
    has_source_section = '## 参考材料原文' in answer
    has_chunk_id = '### chunk-' in answer
    
    print(f"\n【验证结果】")
    print(f"  相关参考部分: {'✅' if has_ref_section else '❌'}")
    print(f"  参考材料原文部分: {'✅' if has_source_section else '❌'}")
    print(f"  chunk编号标记: {'✅' if has_chunk_id else '❌'}")
    
    # 验证citations
    citations = data.get('citations', [])
    print(f"\n  Citations数量: {len(citations)}")
    
    if citations:
        c = citations[0]
        print(f"\n【第一个Citation】")
        print(f"  文档名: {c.get('doc_name', 'N/A')[:50]}")
        print(f"  issuer字段: {c.get('issuer', 'N/A')}")
        print(f"  issue_date字段: {c.get('issue_date', 'N/A')}")
    
    # 显示answer后部分（chunk引用区域）
    if '## 参考材料原文' in answer:
        chunk_section = answer.split('## 参考材料原文')[1][:500]
        print(f"\n【参考材料原文部分预览】")
        print(chunk_section)
    else:
        print(f"\n【Answer前500字符】")
        print(answer[:500])

except Exception as e:
    print(f"错误: {e}")