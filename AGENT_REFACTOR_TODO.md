# Agent框架改造清单（简洁版）

## 一、待完成项

### Phase 1: 意图分类升级 ⏳
| 待完成 | 文件 | 内容 |
|--------|------|------|
| ❌ | intent_rules.py | 规则分类引擎（关键词+置信度计算） |
| ❌ | intent_llm.py | LLM精确分类器（Prompt模板+结果解析） |
| ⚠️ | intent_classifier.py | 重构：双层分类入口（规则→LLM fallback） |
| ❌ | intent_validator.py | 分类结果验证与修正 |

### Phase 2: 数据源集成 ⏳
| 待完成 | 文件 | 内容 |
|--------|------|------|
| ❌ | ~/.electricity_data_skills.json | Skills数据库配置文件 |
| ⚠️ | SkillsScriptAdapter | 完善fetch_sync、命令构建、结果处理 |
| ❌ | data_cache.py | 内存缓存（TTL过期+LRU淘汰） |
| ❌ | mock_adapter.py | Mock数据生成器（测试/降级用） |

### Phase 3: 错误处理与重试 ⏳
| 待完成 | 文件 | 内容 |
|--------|------|------|
| ❌ | error_handler.py | 错误分类（Network/Database/LLM/Timeout） |
| ❌ | retry_handler.py | 重试策略配置（延迟模式+次数） |
| ⚠️ | data_fetcher.py | 集成ErrorHandler+降级逻辑 |
| ⚠️ | policy_retriever.py | 集成重试+web_search fallback |

### Phase 4: 多步规划与执行 ⏳
| 待完成 | 文件 | 内容 |
|--------|------|------|
| ❌ | planner.py | Hybrid场景规划（LLM生成执行计划） |
| ❌ | execution_loop.py | 循环执行器（逐步执行+状态更新） |
| ❌ | context_aggregator.py | 多源结果聚合（policy+data+analysis） |
| ⚠️ | electricity_agent_graph.py | 添加hybrid路由+执行循环节点 |

### Phase 5: 测试覆盖 ⏳
| 待完成 | 文件 | 内容 |
|--------|------|------|
| ❌ | tests/agent/ | 单元测试目录（intent/data/error/loop） |
| ❌ | integration/ | 集成测试（full_flow/hybrid_query/error_recovery） |
| ❌ | fixtures/ | Mock数据（mock_data/mock_llm/mock_adapter） |

---

## 二、主要内容

### 意图分类（双层策略）
```
用户查询 → RuleClassifier（关键词）
    ↓ confidence >= 0.85 → 直接返回
    ↓ confidence < 0.85 → LLMClassifier
    ↓ 返回: intent + sub_intents + confidence + plan
```

### 数据获取（多源+降级）
```
SkillsScriptAdapter → uv run python scripts/run_basic_stats.py
    ↓ 成功 → 返回数据 + 存缓存
    ↓ 失败 → 重试(3次) → 缓存fallback → Mock fallback
```

### 错误处理（分类+重试+降级）
```
Exception → ErrorHandler.classify()
    ↓ NETWORK → 重试3次 → 缓存降级
    ↓ DATABASE → 重试2次 → Mock降级  
    ↓ LLM → 重试1次 → 模板响应
    ↓ TIMEOUT → 重试2次 → 部分结果
```

### Hybrid执行（规划+循环）
```
Hybrid Intent → PlannerNode → 生成计划[Step1, Step2, ...]
    ↓ ExecutionLoop → for step in plan:
        ↓ execute_step(action)
        ↓ update_state()
        ↓ if error → retry_or_fallback()
```

---

## 三、目标框架

### 目标架构图
```
API → AgentSingleton → LangGraph
    ↓
IntentClassifier(LLM+规则) → 路由
    ↓
┌─────────────┬──────────────┬─────────────┐
│ policy      │ data         │ hybrid      │
│ ↓           │ ↓            │ ↓           │
│ PolicyRetriever│ DataFetcher │ Planner    │
│ (RAG+重试)  │ (Skills+缓存)│ ↓           │
│ ↓           │ ↓            │ ExecutionLoop│
│             │ DataAnalyzer │ (循环执行)  │
│             │ (NumPy统计) │ ↓           │
└─────────────┴──────────────┴─────────────┘
    ↓           ↓              ↓
ResponseGenerator(LLM+引用聚合)
    ↓
ConfidenceCalculator → OutputFormatter → 返回
```

### 目标State（扩展版）
```python
ElectricityAgentState:
  # 输入
  query, provinces, messages
  
  # 意图
  intent, sub_intents, intent_confidence, intent_reason
  
  # 规划
  plan: [Step1, Step2, ...], current_step, max_steps
  
  # 检索
  policy_chunks, policy_retrieval_quality, policy_retry_count
  
  # 数据
  electricity_data, data_source, data_fetch_errors
  
  # 分析
  analysis_result, analysis_type, analysis_metrics
  
  # 输出
  answer, citations, confidence, tool_calls, execution_trace
  
  # 元数据
  metadata, errors: [Error1, ...], warnings
```

### 目标成熟度：8/10
| 维度 | 目标 | 当前 |
|------|------|------|
| 意图分类 | LLM+规则双层 | 关键词规则 |
| 流程编排 | 动态规划+循环 | 固定路由 |
| 错误处理 | 重试+降级 | 单次失败 |
| 数据连接 | Skills+缓存 | 未配置 |
| 测试覆盖 | >80% | 0% |
| 响应延迟 | P95<3s | 未优化 |

---

## 四、快速启动步骤

### Step 1: 配置数据库
```bash
# 创建配置文件
python data/skills/agentic-data-analysis/scripts/config.py create-config \
  --host YOUR_HOST --user YOUR_USER --password YOUR_PASSWORD
```

### Step 2: 测试数据获取
```bash
cd data/skills/agentic-data-analysis
uv run python scripts/run_basic_stats.py \
  --region shaanxi --start-date 2026-05-13 --end-date 2026-05-13 \
  --source electricity --tables trading --fields demand
```

### Step 3: 验证Agent
```python
from app.agent.agent_singleton import agent_singleton
from app.config import settings

agent = agent_singleton.preload(settings)
result = agent.chat(AgentRequest(query="陕西昨日负荷", ...))
```

---

**预计工期**: 8周  
**优先级**: Phase1(意图) + Phase2(数据) = 最高