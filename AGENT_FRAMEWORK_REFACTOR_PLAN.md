# Agent框架完整改造方案

## 一、现状评估

### 1.1 当前架构分析

```
用户查询 → AgentSingleton → 框架选择 → ElectricityAgentGraph/PowerPolicyAgent
                                        ↓
                              LangGraph流程（固定路由）：
                              intent_classifier(关键词) → 路由 → policy_retriever/data_fetcher → response_generator
                              
问题：固定路由无法动态组合工具，Agent无自主决策能力
```

### 1.2 成熟度评估

| 维度 | 当前状态 | 成熟标准 | 差距 |
|------|----------|----------|------|
| **架构模式** | StateGraph固定路由 | ReAct循环动态决策 | 高 |
| **工具选择** | 预定义组合 | Agent自主选择 | 高 |
| **意图识别** | 关键词规则 | Agent自主判断 + 规则辅助 | 高 |
| **错误处理** | 单次失败返回 | 重试 + fallback + 降级 | 高 |
| **状态管理** | MemorySaver未用 | Checkpoint持久化 | 中 |
| **数据连接** | 未配置 | Skills数据库连接 | 高 |
| **思考链** | 无 | 完整思考过程记录 | 高 |
| **可观测性** | 基础日志 | LangSmith + 结构化日志 | 中 |
| **测试覆盖** | 0% | >80% | 高 |

**整体评分**: 3/10（原型阶段）

**目标**: 升级为ReAct Agent架构，实现Agent自主决策、动态工具组合、循环执行能力

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

### 2.1 ReAct Agent框架设计

**核心原理**：LLM观察当前状态，自主思考并决定下一步行动，循环执行直到完成任务。

```
入口层:
  API Router → AgentSingleton → ReActAgent

ReAct核心循环:
  用户查询 → Agent思考(Think)
              ↓
          决策: 使用工具 / 直接回答 / 继续思考
              ↓
          若使用工具: 执行Tool → 观察结果 → 继续思考
              ↓
          若回答: 返回最终答案

工具层（Tool定义）:
  @tool retrieve_policy(query, provinces)     → 政策检索
  @tool fetch_electricity_data(province, metric, time_range) → 数据获取
  @tool analyze_statistics(data, type)        → 统计分析
  @tool web_search(query)                     → 网络搜索
  @tool calculate_expression(expr)            → 数值计算

数据层（多源适配）:
  SkillsScriptAdapter → 数据库
  SkillsAPIAdapter    → HTTP API
  LocalDataAdapter    → 本地文件
  CacheAdapter        → Redis缓存

基础设施:
  ToolExecutor(工具执行) + ErrorHandler(重试+降级) + ThoughtChain(思考记录) + LangSmith(追踪)
```

### 2.2 ReAct执行流程示例

```
用户: "陕西昨天负荷均值是多少，对比政策规定"

循环1:
  Think: 需要获取负荷数据并分析，还要检索政策
  Action: fetch_electricity_data(province="SN", metric="load", time_range="24h")
  Observe: 获得96点负荷数据
  
循环2:
  Think: 数据已获取，需要计算均值
  Action: analyze_statistics(data, type="mean")
  Observe: 均值=3250MW
  
循环3:
  Think: 需要检索负荷相关政策
  Action: retrieve_policy(query="负荷规定", provinces=["SN"])
  Observe: 检索到3个政策片段
  
循环4:
  Think: 信息已完整，可以回答
  Answer: 陕西昨日负荷均值3250MW，政策规定...
```

### 2.3 目标State定义

```python
class ElectricityAgentState(TypedDict):
    # 输入层
    messages: Annotated[List[Dict], add_messages]
    query: str
    provinces: List[str]
    
    # 思考链（ReAct核心）
    thoughts: List[Dict]           # [{"thought": "...", "action": "...", "tool": "...", "args": {...}}]
    iteration_count: int           # 当前迭代次数
    max_iterations: int            # 最大迭代限制（防止无限循环）
    
    # 工具调用记录
    tool_calls: List[str]          # 调用的工具名称列表
    tool_results: List[Dict]       # 工具返回结果
    
    # 检索层
    policy_chunks: List[Dict]
    policy_retrieval_quality: Dict
    
    # 数据层
    electricity_data: Optional[Dict]
    data_source: str
    
    # 分析层
    analysis_result: Optional[Dict]
    
    # 输出层
    answer: str
    citations: List[Dict]
    confidence: float
    done: bool                     # 是否完成
    
    # 元数据
    metadata: Dict[str, Any]
    errors: List[Dict]
    execution_trace: List[Dict]    # 完整执行轨迹
```

---

## 三、详细改造方案

### 3.0 Phase 0: Tool化改造（优先级：最高）

**目标**: 将现有节点转换为Tool定义，实现ReAct循环机制

**核心改动**:

1. **创建tools目录，定义Tool接口**

新增文件结构:
```
app/agent/tools/
├── __init__.py
├── policy_tool.py      # 政策检索Tool
├── data_tool.py        # 数据获取Tool  
├── analysis_tool.py    # 数据分析Tool
├── web_tool.py         # 网络搜索Tool
└── tool_executor.py    # Tool执行器
```

Tool定义示例（不写完整代码）:
```
@tool装饰器定义各工具:
- retrieve_policy: query + provinces → 政策片段列表
- fetch_electricity_data: province + metric + time_range → 数据字典
- analyze_statistics: data + type → 统计结果
- web_search: query → 搜索结果
```

2. **实现ReAct核心循环**

新增文件:
```
app/agent/graph/
├── react_agent.py      # ReAct循环核心
├── tool_node.py        # ToolNode实现
└── thought_chain.py    # 思考链管理
```

ReAct循环逻辑:
```
while iteration < max_iterations:
    1. LLM观察当前状态（query + history + tool_results）
    2. LLM思考并决策：
       - 若有tool_calls → 执行工具 → 记录结果 → 继续
       - 若无tool_calls → 生成答案 → done=True → 结束
    3. 记录思考链（thoughts字段）
    4. 迭代次数+1
```

3. **修改LangGraph图结构**

从固定路由改为ReAct循环:
```
原架构:
  intent_classifier → 路由 → policy_retriever/data_fetcher → response_generator

新架构:
  react_agent ←→ tool_node（循环）
  react_agent: LLM思考决策
  tool_node: 执行工具并返回结果
  条件路由: tool_calls存在则执行工具，否则结束
```

### 3.1 Phase 1: Tool层完善（优先级：高）

**目标**: 完善所有Tool定义，支持Agent自主调用

**Tool列表**:

| Tool名称 | 功能 | 输入参数 | 输出 |
|----------|------|----------|------|
| retrieve_policy | 政策检索 | query, provinces | policy_chunks |
| fetch_electricity_data | 数据获取 | province, metric, time_range | electricity_data |
| analyze_statistics | 统计分析 | data, analysis_type | analysis_result |
| web_search | 网络搜索 | query | search_result |
| calculate | 数值计算 | expression | result |

**Tool实现要点**:
- 每个Tool包含清晰的docstring（供LLM理解用途）
- 参数类型明确（供LLM正确调用）
- 错误处理内置（返回友好错误信息）
- 结果格式统一（便于Agent理解）

### 3.2 Phase 2: 数据源集成（优先级：高）

**目标**: 完整集成Skills数据库，实现真实数据查询

（保持原内容不变，数据层改动不受架构影响）

### 3.3 Phase 3: 错误处理与重试（优先级：中）

**目标**: 完善错误处理，支持Tool执行失败时的重试和降级

**Tool级别错误处理**:
```
Tool执行失败 → 
  1. 记录错误到errors字段
  2. 返回友好错误信息给Agent
  3. Agent决定：重试 / 使用替代工具 / 降级回答
```

### 3.4 Phase 4: 思考链与迭代控制（优先级：中）

**目标**: 完善思考链记录，防止无限循环

**思考链结构**:
```python
thought_entry = {
    "iteration": 1,
    "thought": "需要获取负荷数据进行统计分析",
    "action": "use_tool",
    "tool": "fetch_electricity_data",
    "args": {"province": "SN", "metric": "load"},
    "result": "成功获取96点数据",
    "timestamp": "..."
}
```

**迭代控制机制**:
```
max_iterations = 5  # 默认最大迭代次数
iteration_count监控:
  - 若达到max_iterations → 强制结束，返回部分结果
  - 若连续2次相同tool_calls → 检测循环，强制结束
  - 若done=True → 正常结束
```

### 3.5 Phase 5: 测试覆盖（优先级：高）

**目标**: 编写完整单元测试，覆盖率>80%

**测试文件结构**:
```
tests/agent/
├── test_react_agent.py          # ReAct循环测试
├── test_tools.py                # Tool定义测试
├── test_tool_executor.py        # 工具执行测试
├── test_thought_chain.py        # 思考链测试
├── test_policy_tool.py          # 政策检索测试
├── test_data_tool.py            # 数据获取测试
├── test_analysis_tool.py        # 数据分析测试
├── test_error_handler.py        # 错误处理测试
├── integration/
│   ├── test_full_flow.py        # 完整流程测试
│   ├── test_multi_tool.py       # 多工具组合测试
│   └── test_iteration_limit.py  # 迭代限制测试
└── fixtures/
    ├── mock_tools.py
    ├── mock_llm.py
    └── mock_adapter.py
```

**关键测试场景**:
```
- 单工具调用：验证Tool正确执行
- 多工具组合：验证Agent自主组合能力
- 迭代限制：验证max_iterations生效
- 错误恢复：验证Tool失败后的处理
- 思考链完整：验证thoughts字段记录完整
```

---

## 四、实施路线图

### 4.1 时间规划（10周）

| 周 | 任务 | 交付物 |
|----|------|--------|
| W1 | Phase 0: Tool化改造 | tools目录、Tool定义、ToolNode |
| W2 | Phase 0: ReAct循环实现 | react_agent.py、思考链机制 |
| W3 | Phase 1: Tool层完善 | 所有Tool定义完成、docstring清晰 |
| W4 | Phase 2: 数据源集成 | SkillsScriptAdapter完善、数据库配置 |
| W5 | Phase 3: 错误处理 | ErrorHandler、Tool级别重试 |
| W6 | Phase 4: 思考链优化 | 完整思考链记录、迭代控制 |
| W7 | Phase 5: 核心测试 | 单元测试覆盖率>60% |
| W8 | Phase 5: 集成测试 | 完整流程测试、多工具组合测试 |
| W9 | 文档完善 | API文档、Tool使用文档 |
| W10 | 最终验收 | 性能测试、部署验证 |

### 4.2 验收标准

| 维度 | 目标 | 验收方式 |
|------|------|----------|
| ReAct循环成功率 | >90% | 多场景测试 |
| 工具组合正确率 | >85% | Agent自主组合测试 |
| 思考链完整度 | 100% | 检查thoughts字段 |
| 迭代限制生效 | 100% | max_iterations测试 |
| 数据获取成功率 | >95% | Mock+真实测试 |
| 测试覆盖率 | >80% | pytest --cov |
| 响应延迟 | P95<30s（含多次迭代） | 性能测试 |

---

## 五、配置与部署

### 5.1 配置文件完善

```python
# app/config.py 新增配置
@dataclass
class Settings:
    # ReAct Agent框架
    agent_framework: str = "react"          # 改为react模式
    agent_max_iterations: int = 5           # 最大迭代次数
    agent_tool_timeout: int = 30            # Tool执行超时
    
    # Tool配置
    tools_enabled: List[str] = [
        "retrieve_policy",
        "fetch_electricity_data",
        "analyze_statistics",
        "web_search",
    ]
    
    # 数据源
    electricity_skills_path: str = "./data/skills/agentic-data-analysis"
    electricity_db_host: str = ""
    electricity_db_port: int = 3306
    electricity_db_user: str = ""
    electricity_db_password: str = ""
    
    # 缓存
    data_cache_ttl: int = 3600
    data_cache_max_size: int = 1000
    
    # LangSmith追踪
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "electricity-agent"
```

### 5.2 .env完整配置

```bash
# ReAct Agent框架
AGENT_FRAMEWORK=react
AGENT_MAX_ITERATIONS=5
AGENT_TOOL_TIMEOUT=30

# 启用的工具
TOOLS_ENABLED=retrieve_policy,fetch_electricity_data,analyze_statistics,web_search

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
REACT_ITERATION_COUNT = Counter('agent_iterations_total', 'Total iterations', ['session'])
TOOL_CALL_COUNT = Counter('agent_tool_calls_total', 'Tool calls', ['tool', 'status'])
TOOL_LATENCY = Histogram('agent_tool_latency', 'Tool execution time', ['tool'])
THOUGHT_LENGTH = Histogram('agent_thought_length', 'Thought chain length')
ERROR_COUNT = Counter('agent_error_total', 'Error count', ['tool', 'category'])
```

### 6.2 健康检查

```python
def agent_health_check() -> Dict:
    checks = {
        "agent_loaded": agent_singleton.is_loaded(),
        "framework": agent_singleton.get_framework(),
        "tools_available": len(get_available_tools()),
        "db_configured": adapter._db_configured,
        "tool_test": "success/failed",
        "llm_test": "success/failed",
    }
    checks["healthy"] = all(v in [True, "success"] for v in checks.values())
    return checks
```

---

## 七、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Agent无限循环 | 资源耗尽 | max_iterations限制 + 循环检测 |
| LLM决策不稳定 | 工具调用错误 | Tool docstring清晰 + 参数校验 |
| Skills数据库不可用 | 数据获取失败 | 多数据源fallback + 缓存 + Mock |
| 网络超时 | 响应延迟 | Tool超时配置 + 重试 + 降级 |
| Token消耗过大 | 成本升高 | 迭代限制 + 思考链压缩 |

---

## 八、改造后文件清单

```
app/agent/
├── agent_singleton.py              # 重构：支持react框架
├── graph/
│   ├── react_agent.py              # 新增：ReAct核心循环
│   ├── tool_node.py                # 新增：ToolNode实现
│   ├── thought_chain.py            # 新增：思考链管理
│   ├── state.py                    # 重构：添加thoughts等字段
│   ├── handlers/
│   │   ├── error_handler.py        # 新增
│   │   └── retry_handler.py        # 新增
│   └── tracing.py                  # 新增：LangSmith
├── tools/
│   ├── __init__.py                 # 新增
│   ├── policy_tool.py              # 新增：政策检索Tool
│   ├── data_tool.py                # 新增：数据获取Tool
│   ├── analysis_tool.py            # 新增：数据分析Tool
│   ├── web_tool.py                 # 新增：网络搜索Tool
│   └── tool_executor.py            # 新增：工具执行器
├── adapters/
│   ├── electricity_data_adapter.py # 保持
│   ├── data_cache.py               # 新增
│   └── mock_adapter.py             # 新增
├── metrics.py                      # 新增
└── health.py                       # 新增

tests/agent/
├── test_react_agent.py             # 新增
├── test_tools.py                   # 新增
├── test_tool_executor.py           # 新增
├── test_thought_chain.py           # 新增
├── integration/
│   ├── test_full_flow.py
│   └── test_multi_tool.py          # 新增
└── fixtures/
    ├── mock_tools.py
    └── mock_llm.py
```

---

## 九、与原架构对比

| 维度 | 原架构（StateGraph） | 新架构（ReAct） |
|------|---------------------|-----------------|
| 流程控制 | 固定路由 | Agent自主决策 |
| 工具选择 | 预定义组合 | 动态选择组合 |
| 复杂查询 | Planner一次性规划 | 循环迭代执行 |
| 错误处理 | 静态降级 | Agent动态调整 |
| 可扩展性 | 需修改路由逻辑 | 新增Tool即可 |
| 可观测性 | 节点日志 | 完整思考链 |

---

**文档版本**: v2.0（ReAct架构升级版）  
**创建日期**: 2026-05-14  
**预计完成**: 2026-07-21（10周）  
**成熟度目标**: 从3/10提升至8/10（生产可用）  
**核心改动**: 从固定路由StateGraph升级为ReAct动态决策Agent