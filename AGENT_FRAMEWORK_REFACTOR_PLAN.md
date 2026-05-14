# Agent框架完整改造方案

## 一、现状评估

### 1.1 当前架构分析

```
用户查询 → AgentSingleton → 框架选择 → ElectricityAgentGraph/PowerPolicyAgent
                                        ↓
                              LangGraph流程：
                              intent_classifier(关键词) → 路由 → policy_retriever/data_fetcher → response_generator
```

### 1.2 成熟度评估

| 维度 | 当前状态 | 成熟标准 | 差距 |
|------|----------|----------|------|
| **意图识别** | 关键词规则 | LLM分类 + 规则fallback | 高 |
| **流程编排** | 固定路由 | 动态规划 + 循环执行 | 高 |
| **错误处理** | 单次失败返回 | 重试 + fallback + 降级 | 高 |
| **状态管理** | MemorySaver未用 | Checkpoint持久化 | 中 |
| **数据连接** | 未配置 | Skills数据库连接 | 高 |
| **可观测性** | 基础日志 | LangSmith + 结构化日志 | 中 |
| **测试覆盖** | 0% | >80% | 高 |

**整体评分**: 3/10（原型阶段）

### 1.3 现有代码资产

| 文件 | 行数 | 可复用度 | 说明 |
|------|------|----------|------|
| electricity_agent_graph.py | 207 | 高 | StateGraph核心结构 |
| state.py | 16 | 高 | State定义完整 |
| intent_classifier.py | 37 | 低 | 仅关键词，需重构 |
| policy_retriever.py | 89 | 高 | 复用Orchestrator |
| data_fetcher.py | 134 | 中 | 结构完整，数据源待连接 |
| data_analyzer.py | 90 | 中 | 基础统计，需扩展 |
| response_generator.py | 127 | 高 | LLM调用完整 |

---

## 二、目标架构

### 2.1 成熟Agent框架设计

```
入口层:
  API Router → AgentSingleton → 框架选择(langgraph/react)

LangGraph核心:
  IntentClassifier(LLM+规则双层) → 路由决策
    ↓
  policy: PolicyRetriever(RAG+重试) → ContextAggregator
  data: DataFetcher(多源+降级) → DataAnalyzer(统计+预测) → ContextAggregator
  hybrid: PlannerNode → ExecutionLoop → 循环执行
    ↓
  ResponseGenerator(LLM+引用) → ConfidenceCalculator → OutputFormatter

数据层:
  SkillsScriptAdapter → 数据库
  SkillsAPIAdapter → HTTP API
  LocalDataAdapter → 本地文件
  CacheAdapter → Redis缓存

基础设施:
  CheckpointManager(持久化) + ErrorHandler(重试+降级) + TraceService(LangSmith)
```

### 2.2 目标State定义

```python
class ElectricityAgentState(TypedDict):
    # 输入层
    messages: Annotated[List[Dict], add_messages]
    query: str
    provinces: List[str]
    
    # 意图层
    intent: str
    sub_intents: List[str]
    intent_confidence: float
    intent_reason: str
    
    # 规划层
    plan: List[Dict]
    current_step: int
    max_steps: int
    
    # 检索层
    policy_chunks: List[Dict]
    policy_retrieval_quality: Dict
    policy_retry_count: int
    
    # 数据层
    electricity_data: Optional[Dict]
    data_source: str
    data_fetch_errors: List[str]
    
    # 分析层
    analysis_result: Optional[Dict]
    analysis_type: str
    analysis_metrics: Dict
    
    # 输出层
    answer: str
    citations: List[Dict]
    confidence: float
    tool_calls: List[str]
    execution_trace: List[Dict]
    
    # 元数据
    metadata: Dict[str, Any]
    errors: List[Dict]
    warnings: List[str]
```

---

## 三、详细改造方案

### 3.1 Phase 1: 意图分类升级（优先级：高）

**目标**: 从关键词规则升级为LLM+规则双层分类

**双层分类策略**:
```
Level 1: 快速规则分类（关键词匹配） → confidence >= 0.85 → 直接返回
    ↓
Level 2: LLM精确分类（不确定时触发） → confidence < 0.85 → LLM分类
    ↓
输出: intent + sub_intents + confidence + reason + plan
```

**关键词分类规则**:
| Intent | 关键词 | 置信度阈值 |
|--------|--------|------------|
| policy_query | 政策、规则、通知、规定、条款、准入 | 0.9（直接返回） |
| data_query | 负荷、发电量、用电量、电价、实时、曲线 | 0.9（直接返回） |
| analysis | 统计、均值、方差、分析、趋势、增长、分布 | 0.9（直接返回） |
| hybrid | 混合关键词/不确定 | <0.7（触发LLM） |

**新增文件**:
```
app/agent/graph/nodes/
├── intent_classifier.py      # 重构：双层分类入口
├── intent_rules.py           # 新增：规则引擎
├── intent_llm.py             # 新增：LLM分类器
└── intent_validator.py       # 新增：分类验证
```

**核心代码 - intent_rules.py**:
```python
class RuleClassifier:
    POLICY_KEYWORDS = {
        "primary": ["政策", "规则", "通知", "规定", "条款", "准入"],
        "secondary": ["办法", "细则", "指南", "要求"]
    }
    
    DATA_KEYWORDS = {
        "primary": ["负荷", "发电量", "用电量", "电价", "实时", "曲线"],
        "secondary": ["数据", "功率", "出力", "现货"]
    }
    
    ANALYSIS_KEYWORDS = {
        "primary": ["统计", "均值", "方差", "分析", "趋势", "增长"],
        "secondary": ["分布", "对比", "预测", "评估"]
    }
    
    @classmethod
    def classify(cls, query: str) -> Dict:
        scores = cls._score_intent(query)
        primary_intent = cls._determine_primary(scores)
        confidence = cls._calculate_confidence(scores)
        
        return {
            "intent": primary_intent,
            "confidence": confidence,
            "reason": cls._generate_reason(scores)
        }
    
    @classmethod
    def _calculate_confidence(cls, scores: Dict) -> float:
        max_score = max(scores.values())
        high_score_count = sum(1 for s in scores.values() if s >= 1)
        
        if high_score_count == 1:
            return min(0.95, max_score * 0.3)  # 单一意图高置信度
        elif high_score_count > 1:
            return 0.6  # 多意图混合低置信度
        return min(0.8, max_score * 0.25)
```

### 3.2 Phase 2: 数据源集成（优先级：高）

**目标**: 完整集成Skills数据库，实现真实数据查询

**配置方式**:
```json
// ~/.electricity_data_skills.json
{
  "host": "数据库主机",
  "port": 3306,
  "user": "用户名",
  "password": "密码",
  "base_db_name": "electricity_trading_analytics",
  "region_db_prefix": "electricity_trading_analytics"
}
```

**或环境变量**:
```bash
DB_HOST=localhost
DB_PORT=3306
DB_USER=用户名
DB_PASSWORD=密码
```

**SkillsScriptAdapter支持的指标**:
| Metric | 表名 | 字段 |
|--------|------|------|
| load | trading | demand |
| generation | trading | renewable_output |
| pv | trading | pv_output |
| wind | trading | wind_output |
| rtcp | clearing_price | realtime_clearing_price |
| dacp | clearing_price | dayahead_clearing_price |

**新增DataCache实现**:
```python
class DataCache:
    """数据缓存器 - TTL过期 + LRU淘汰"""
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self._cache: Dict[str, Dict] = {}
        self._ttl = ttl
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        entry = self._cache[key]
        if time.time() - entry["timestamp"] > self._ttl:
            del self._cache[key]
            return None
        return entry["data"]
    
    def set(self, key: str, data: Any) -> None:
        # LRU淘汰
        if len(self._cache) >= self._max_size:
            oldest = min(self._cache.items(), key=lambda x: x[1]["timestamp"])
            del self._cache[oldest[0]]
        self._cache[key] = {"data": data, "timestamp": time.time()}
```

### 3.3 Phase 3: 错误处理与重试（优先级：中）

**目标**: 完善错误处理，实现自动重试和降级

**错误分类**:
| Category | 触发条件 | 重试策略 | 降级方案 |
|----------|----------|----------|----------|
| NETWORK | 连接失败 | 3次，延迟1/3/5秒 | 缓存数据 |
| DATABASE | SQL错误 | 2次，延迟2/5秒 | Mock数据 |
| LLM | API超时 | 1次，延迟3秒 | 模板响应 |
| TIMEOUT | 执行超时 | 2次，延迟5/10秒 | 部分结果 |

**新增ErrorHandler**:
```python
class ErrorHandler:
    RETRY_STRATEGIES = {
        ErrorCategory.NETWORK: {
            "max_retries": 3,
            "delay_pattern": [1, 3, 5],
            "backoff": "exponential"
        },
        ErrorCategory.DATABASE: {
            "max_retries": 2,
            "delay_pattern": [2, 5],
            "backoff": "linear"
        },
        ErrorCategory.LLM: {
            "max_retries": 1,
            "delay_pattern": [3],
            "backoff": "fixed"
        }
    }
    
    FALLBACK_STRATEGIES = {
        ErrorCategory.NETWORK: "cache",
        ErrorCategory.DATABASE: "mock_data",
        ErrorCategory.LLM: "template_response",
    }
    
    @classmethod
    def handle_with_retry(cls, func, *args, **kwargs) -> tuple:
        strategy = kwargs.pop("retry_strategy", None)
        for attempt in range(strategy["max_retries"] + 1):
            try:
                result = func(*args, **kwargs)
                return (True, result, None)
            except Exception as e:
                if attempt == strategy["max_retries"]:
                    return (False, None, e)
                delay = cls._calculate_delay(strategy, attempt)
                time.sleep(delay)
        return (False, None, None)
```

### 3.4 Phase 4: 多步规划与执行循环（优先级：中）

**目标**: 实现hybrid场景的多步骤规划与循环执行

**Hybrid执行流程**:
```
用户: "对比陕西昨日负荷与政策规则"
    ↓
PlannerNode生成计划:
  Step 1: retrieve_policy(query="负荷规则")
  Step 2: fetch_data(metric="load", province="SN")
  Step 3: analyze(type="statistical")
  Step 4: aggregate(policy_chunks + analysis_result)
  Step 5: generate_response()
    ↓
ExecutionLoop循环执行:
  for step in plan:
    execute_step(action, params)
    update_state()
    if error: retry_or_fallback()
    if complete: break
```

**新增PlannerNode**:
```python
def planner_node(state: ElectricityAgentState) -> Dict:
    query = state["query"]
    intent = state.get("intent")
    
    if intent != "hybrid":
        return {"plan": _default_plan(intent)}
    
    # Hybrid场景使用LLM规划
    llm_wrapper = _get_llm_wrapper()
    prompt = PLANNER_PROMPT.format(query=query, ...)
    
    response = llm_wrapper.invoke_text(query, system=prompt)
    plan = json.loads(response)
    plan = _validate_plan(plan)
    
    return {"plan": plan, "current_step": 0, "max_steps": len(plan)}

def execution_loop_node(state: ElectricityAgentState) -> Dict:
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    
    if current_step >= len(plan):
        return {"current_step": current_step}
    
    step = plan[current_step]
    result = _execute_step(step["action"], step["params"], state)
    
    return {"current_step": current_step + 1, **result}
```

### 3.5 Phase 5: 测试覆盖（优先级：高）

**目标**: 编写完整单元测试，覆盖率>80%

**测试文件结构**:
```
tests/agent/
├── test_intent_classifier.py    # 意图分类测试
├── test_policy_retriever.py     # 政策检索测试
├── test_data_fetcher.py         # 数据获取测试
├── test_data_analyzer.py        # 数据分析测试
├── test_response_generator.py   # 响应生成测试
├── test_error_handler.py        # 错误处理测试
├── test_execution_loop.py       # 执行循环测试
├── test_data_adapter.py         # 数据适配器测试
├── integration/
│   ├── test_full_flow.py        # 完整流程测试
│   ├── test_hybrid_query.py     # Hybrid场景测试
│   └── test_error_recovery.py   # 错误恢复测试
└── fixtures/
    ├── mock_data.py
    ├── mock_llm.py
    └── mock_adapter.py
```

**关键测试示例**:
```python
class TestIntentClassifier:
    def test_policy_query_detection(self):
        state = {"query": "陕西省电力交易规则", "provinces": ["SN"]}
        result = intent_classifier_node(state)
        assert result["intent"] == "policy_query"
        assert result["intent_confidence"] >= 0.85
    
    def test_hybrid_query_detection(self):
        state = {"query": "对比陕西昨日负荷与政策规定", "provinces": ["SN"]}
        result = intent_classifier_node(state)
        assert result["intent"] == "hybrid"
        assert len(result.get("plan", [])) > 1

class TestSkillsScriptAdapter:
    def test_province_mapping(self):
        adapter = SkillsScriptAdapter()
        assert adapter._map_province_to_region("SN") == "shaanxi"
        assert adapter._map_province_to_region("SX") == "shanxi"
    
    def test_data_fetch_with_mock(self):
        adapter = SkillsScriptAdapter()
        result = adapter.fetch_sync("SN", "load", "24h")
        assert result["province"] == "SN"
        assert result["metric"] == "load"
```

---

## 四、实施路线图

### 4.1 时间规划（8周）

| 周 | 任务 | 交付物 |
|----|------|--------|
| W1 | 意图分类升级 | intent_rules.py, intent_llm.py |
| W2 | 数据源连接 | SkillsScriptAdapter完善, 数据库配置 |
| W3 | 错误处理 | ErrorHandler, 重试策略, 降级机制 |
| W4 | 多步规划 | PlannerNode, ExecutionLoop |
| W5 | 核心测试 | 单元测试覆盖率>60% |
| W6 | 集成测试 | 完整流程测试, 错误恢复测试 |
| W7 | 文档完善 | API文档, 配置文档 |
| W8 | 最终验收 | 性能测试, 部署验证 |

### 4.2 验收标准

| 维度 | 目标 | 验收方式 |
|------|------|----------|
| 意图分类准确率 | >90% | 测试集验证 |
| 数据获取成功率 | >95% | Mock+真实测试 |
| 错误恢复成功率 | >80% | 错误场景测试 |
| 测试覆盖率 | >80% | pytest --cov |
| 响应延迟 | P95<3s | 性能测试 |

---

## 五、配置与部署

### 5.1 配置文件完善

```python
# app/config.py 新增配置
@dataclass
class Settings:
    # Agent框架
    agent_framework: str = "langgraph"
    agent_max_iterations: int = 5
    agent_retry_max: int = 3
    
    # 意图分类
    intent_rule_threshold: float = 0.85
    intent_llm_threshold: float = 0.70
    
    # 数据源
    electricity_skills_path: str = "./data/skills/agentic-data-analysis"
    electricity_db_host: str = ""
    electricity_db_port: int = 3306
    electricity_db_user: str = ""
    electricity_db_password: str = ""
    
    # 缓存
    data_cache_ttl: int = 3600
    data_cache_max_size: int = 1000
    
    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "electricity-agent"
```

### 5.2 .env完整配置

```bash
# Agent框架
AGENT_FRAMEWORK=langgraph
AGENT_MAX_ITERATIONS=5

# 意图分类
INTENT_RULE_THRESHOLD=0.85
INTENT_LLM_THRESHOLD=0.70

# Skills数据源
ELECTRICITY_SKILLS_PATH=./data/skills/agentic-data-analysis
DB_HOST=your_database_host
DB_PORT=3306
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# 缓存
DATA_CACHE_TTL=3600

# LangSmith
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=electricity-agent
```

---

## 六、监控与运维

### 6.1 监控指标

```python
# Prometheus指标
INTENT_CLASSIFICATION_COUNT = Counter('agent_intent_total', 'Intent count', ['intent', 'method'])
DATA_FETCH_COUNT = Counter('agent_data_fetch_total', 'Data fetch', ['source', 'status'])
EXECUTION_LATENCY = Histogram('agent_execution_time', 'Execution time')
ERROR_COUNT = Counter('agent_error_total', 'Error count', ['category', 'node'])
```

### 6.2 健康检查

```python
def agent_health_check() -> Dict:
    checks = {
        "agent_loaded": agent_singleton.is_loaded(),
        "framework": agent_singleton.get_framework(),
        "db_configured": adapter._db_configured,
        "data_fetch_test": "success/failed",
        "llm_test": "success/failed",
    }
    checks["healthy"] = all(v in [True, "success"] for v in checks.values())
    return checks
```

---

## 七、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Skills数据库不可用 | 数据获取失败 | 多数据源fallback + 缓存 + Mock |
| LLM意图分类不稳定 | 分类准确率低 | 规则fallback + 置信度阈值 |
| 网络超时 | 响应延迟 | 重试策略 + 超时配置 + 降级 |
| 内存占用过高 | 性能下降 | 缓存淘汰 + Checkpoint清理 |

---

## 八、改造后文件清单

```
app/agent/
├── agent_singleton.py              # 重构：双框架
├── graph/
│   ├── electricity_agent_graph.py  # 核心
│   ├── state.py                    # 扩展State
│   ├── nodes/
│   │   ├── intent_classifier.py    # 重构：双层
│   │   ├── intent_rules.py         # 新增
│   │   ├── intent_llm.py           # 新增
│   │   ├── planner.py              # 新增
│   │   ├── execution_loop.py       # 新增
│   │   ├── policy_retriever.py     # 重构：重试
│   │   ├── data_fetcher.py         # 重构：重试
│   │   ├── data_analyzer.py        # 重构：扩展
│   │   ├── response_generator.py   # 重构：引用
│   │   └── context_aggregator.py   # 新增
│   ├── handlers/
│   │   ├── error_handler.py        # 新增
│   │   └── retry_handler.py        # 新增
│   ├── tracing.py                  # 新增：LangSmith
│   └── checkpoint.py               # 新增：持久化
├── adapters/
│   ├── electricity_data_adapter.py # 重构
│   ├── data_cache.py               # 新增
│   └── mock_adapter.py             # 新增
├── metrics.py                      # 新增
├── health.py                       # 新增
└── logging_config.py               # 新增

tests/agent/
├── test_intent_classifier.py
├── test_data_adapter.py
├── test_error_handler.py
├── integration/
│   ├── test_full_flow.py
│   └── test_hybrid_query.py
└── fixtures/
    ├── mock_data.py
    └── mock_llm.py
```

---

**文档版本**: v1.0  
**创建日期**: 2026-05-14  
**预计完成**: 2026-07-10（8周）  
**成熟度目标**: 从3/10提升至8/10（生产可用）