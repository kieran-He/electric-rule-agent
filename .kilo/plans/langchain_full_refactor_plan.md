# LangChain完整重构方案详细设计（方案3）

---

## 用户最终决策：方案3 - 完整LangChain重构

---

## 方案3核心架构

```
FastAPI Router (routes_query.py)
         ↓
QueryService
         ↓
LangChainRAGAgent (完全重构)
    ├─ Components:
    │   ├─ HuggingFaceEmbeddings ("BAAI/bge-small-zh-v1.5")
    │   ├─ Chroma Vector Store (persist_directory="./data/chroma")
    │   ├─ VectorStoreRetriever (search_type="similarity", k=5)
    │   ├─ ChatAnthropic (model="MiniMax-M2.7")
    │   ├─ ConversationBufferMemory (memory_key="history")
    │   └─ AgentTools:
    │       ├─ RetrieverTool (政策检索)
    │       └─ 更多工具（未来）
    ├─ Agent:
    │   └─ StructuredChatAgent (动态决策)
    └─ Output:
        └─ QueryAnswer (适配现有格式)
```

---

## 实施步骤（完整重构）

### Phase 1: 环境准备（1小时）

**安装依赖**：
```bash
pip install langchain langchain-core langchain-anthropic
pip install langchain-chroma chromadb
pip install langchain-huggingface
```

**目录结构**：
```
app/langchain/
    ├─ __init__.py
    ├─ embeddings.py
    ├─ vectorstore.py
    ├─ retriever.py
    ├─ tools.py
    ├─ llm.py
    ├─ prompts.py
    ├─ agent.py
    ├─ orchestrator.py
    └─ converters.py
```

---

### Phase 2-11详细步骤

参见完整计划文件（已包含所有11个Phase的详细实施步骤）

---

## 关键风险与应对

| 风险 | 级别 | 应对 |
|------|------|------|
| ChromaDB数据不兼容 | 高 | 测试验证，准备重新embedding |
| Embedding维度不一致 | 高 | 严格验证512维 |
| Agent响应格式错误 | 中 | 测试+格式转换器 |
| 性能下降 | 高 | 性能对比+Chain fallback |

---

## 成功标准

**功能**：
- ✅ Agent正常工作
- ✅ ChromaDB数据兼容
- ✅ QueryAnswer格式正确

**性能**：
- ✅ latency ≤ 20秒
- ✅ recall@3 ≥ 0.9

**Ragas**：
- ✅ faithfulness ≥ 0.7（真实值）

---

## 时间预估

**总计**：20小时（3-4天）

参见计划文件Phase 1-11详细时间分解

---

## 后续扩展

参见计划文件"后续扩展路线图"部分

---

**计划文件位置**：`.kilo/plans/1776755162345-mighty-orchid.md`

**完整方案3详细设计已包含在主计划文件中**。