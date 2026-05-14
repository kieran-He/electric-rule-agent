# Agent架构改造任务清单

## 目标

将StateGraph固定路由架构升级为ReAct动态决策架构，实现Agent自主工具选择能力。

---

## Phase 0: Tool化改造（W1-W2，最高优先级）

### 任务列表

| # | 任务 | 交付物 | 状态 |
|---|------|--------|------|
| 0.1 | 创建tools目录结构 | `app/agent/tools/`目录 | 待开始 |
| 0.2 | 定义政策检索Tool | `policy_tool.py` | 待开始 |
| 0.3 | 定义数据获取Tool | `data_tool.py` | 待开始 |
| 0.4 | 定义数据分析Tool | `analysis_tool.py` | 待开始 |
| 0.5 | 定义网络搜索Tool | `web_tool.py` | 待开始 |
| 0.6 | 实现Tool执行器 | `tool_executor.py` | 待开始 |
| 0.7 | 实现ReAct核心循环 | `react_agent.py` | 待开始 |
| 0.8 | 实现ToolNode | `tool_node.py` | 待开始 |
| 0.9 | 实现思考链管理 | `thought_chain.py` | 待开始 |
| 0.10 | 重构State定义 | 添加thoughts、iteration_count等字段 | 待开始 |
| 0.11 | 重构LangGraph图结构 | 固定路由改为循环结构 | 待开始 |

### Tool定义规范

每个Tool需包含：
- 清晰的docstring（供LLM理解用途）
- 明确的参数类型定义
- 内置错误处理
- 统一的返回格式

---

## Phase 1: Tool层完善（W3，高优先级）

### 任务列表

| # | 任务 | 交付物 | 状态 |
|---|------|--------|------|
| 1.1 | 完善retrieve_policy Tool | 支持多省份查询 | 待开始 |
| 1.2 | 完善fetch_electricity_data Tool | 支持多种指标和时间范围 | 待开始 |
| 1.3 | 完善analyze_statistics Tool | 均值/方差/分布/趋势分析 | 待开始 |
| 1.4 | 完善web_search Tool | 外部搜索集成 | 待开始 |
| 1.5 | 添加calculate Tool | 数值计算能力 | 待开始 |
| 1.6 | Tool文档完善 | 每个Tool的API文档 | 待开始 |

---

## Phase 2: 数据源集成（W4，高优先级）

### 任务列表

| # | 任务 | 交付物 | 状态 |
|---|------|--------|------|
| 2.1 | 配置Skills数据库连接 | `~/.electricity_data_skills.json` | 待开始 |
| 2.2 | 完善SkillsScriptAdapter | 数据库查询逻辑 | 待开始 |
| 2.3 | 实现数据缓存层 | `data_cache.py`（TTL+LRU） | 待开始 |
| 2.4 | 实现Mock数据适配器 | 测试用假数据 | 待开始 |
| 2.5 | 多数据源fallback机制 | 数据库→API→CSV→Mock | 待开始 |

### 数据指标映射

| Metric | 表名 | 字段 |
|--------|------|------|
| load | trading | demand |
| generation | trading | renewable_output |
| pv | trading | pv_output |
| wind | trading | wind_output |
| rtcp | clearing_price | realtime_clearing_price |
| dacp | clearing_price | dayahead_clearing_price |

---

## Phase 3: 错误处理（W5，中优先级）

### 任务列表

| # | 任务 | 交付物 | 状态 |
|---|------|--------|------|
| 3.1 | 实现ErrorHandler | `error_handler.py` | 待开始 |
| 3.2 | Tool级别重试机制 | NETWORK:3次, DATABASE:2次, LLM:1次 | 待开始 |
| 3.3 | 降级策略实现 | cache/mock/template | 待开始 |
| 3.4 | Tool超时配置 | 默认30秒超时 | 待开始 |

### 错误分类与策略

| 类别 | 重试次数 | 延迟模式 | 降级方案 |
|------|----------|----------|----------|
| NETWORK | 3次 | 1/3/5秒 | 缓存数据 |
| DATABASE | 2次 | 2/5秒 | Mock数据 |
| LLM | 1次 | 3秒 | 模板响应 |
| TIMEOUT | 2次 | 5/10秒 | 部分结果 |

---

## Phase 4: 思考链与迭代控制（W6，中优先级）

### 任务列表

| # | 任务 | 交付物 | 状态 |
|---|------|--------|------|
| 4.1 | 思考链完整记录 | thoughts字段结构定义 | 待开始 |
| 4.2 | 迭代次数限制 | max_iterations=5 | 待开始 |
| 4.3 | 循环检测机制 | 连续相同tool_calls检测 | 待开始 |
| 4.4 | 强制结束处理 | 达到限制返回部分结果 | 待开始 |

### 思考链记录结构

每条记录包含：iteration、thought、action、tool、args、result、timestamp

---

## Phase 5: 测试覆盖（W7-W8，高优先级）

### 任务列表

| # | 任务 | 交付物 | 状态 |
|---|------|--------|------|
| 5.1 | ReAct循环测试 | `test_react_agent.py` | 待开始 |
| 5.2 | Tool定义测试 | `test_tools.py` | 待开始 |
| 5.3 | Tool执行器测试 | `test_tool_executor.py` | 待开始 |
| 5.4 | 思考链测试 | `test_thought_chain.py` | 待开始 |
| 5.5 | 各Tool单元测试 | test_policy/data/analysis_tool.py | 待开始 |
| 5.6 | 错误处理测试 | `test_error_handler.py` | 待开始 |
| 5.7 | 完整流程集成测试 | `integration/test_full_flow.py` | 待开始 |
| 5.8 | 多工具组合测试 | `integration/test_multi_tool.py` | 待开始 |
| 5.9 | 迭代限制测试 | `integration/test_iteration_limit.py` | 待开始 |
| 5.10 | Mock fixtures | fixtures/mock_tools/llm/adapter.py | 待开始 |

### 测试覆盖目标

单元测试覆盖率 > 80%
集成测试覆盖核心场景

---

## Phase 6: 文档与部署（W9-W10）

### 任务列表

| # | 任务 | 交付物 | 状态 |
|---|------|--------|------|
| 6.1 | Tool使用文档 | 各Tool的调用示例 | 待开始 |
| 6.2 | Agent配置文档 | .env配置说明 | 待开始 |
| 6.3 | API接口文档 | OpenAPI规范 | 待开始 |
| 6.4 | 性能测试报告 | P95延迟<30秒 | 待开始 |
| 6.5 | 部署验证 | 生产环境可用 | 待开始 |

---

## 文件清单

### 新增文件

```
app/agent/tools/
├── __init__.py
├── policy_tool.py
├── data_tool.py
├── analysis_tool.py
├── web_tool.py
└── tool_executor.py

app/agent/graph/
├── react_agent.py
├── tool_node.py
├── thought_chain.py
├── handlers/
│   ├── error_handler.py
│   └── retry_handler.py
└── tracing.py

app/agent/adapters/
├── data_cache.py
└── mock_adapter.py

tests/agent/
├── test_react_agent.py
├── test_tools.py
├── test_tool_executor.py
├── test_thought_chain.py
├── integration/
│   ├── test_full_flow.py
│   └── test_multi_tool.py
│   └── test_iteration_limit.py
└── fixtures/
    ├── mock_tools.py
    ├── mock_llm.py
    └── mock_adapter.py
```

### 修改文件

```
app/agent/agent_singleton.py    → 支持react框架
app/agent/graph/state.py        → 添加thoughts等字段
app/config.py                   → 新增react配置项
```

---

## 配置清单

### 新增配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| agent_framework | react | 框架模式 |
| agent_max_iterations | 5 | 最大迭代次数 |
| agent_tool_timeout | 30 | Tool超时秒数 |
| tools_enabled | retrieve_policy,fetch_electricity_data,... | 启用工具列表 |
| data_cache_ttl | 3600 | 缓存过期秒数 |

---

## 验收标准

| 维度 | 目标 |
|------|------|
| ReAct循环成功率 | >90% |
| 工具组合正确率 | >85% |
| 思考链完整度 | 100% |
| 迭代限制生效 | 100% |
| 数据获取成功率 | >95% |
| 测试覆盖率 | >80% |
| 响应延迟P95 | <30秒 |

---

## 时间规划

| 周 | Phase | 重点 |
|----|-------|------|
| W1-W2 | Phase 0 | Tool化改造、ReAct循环 |
| W3 | Phase 1 | Tool完善 |
| W4 | Phase 2 | 数据源集成 |
| W5 | Phase 3 | 错误处理 |
| W6 | Phase 4 | 思考链控制 |
| W7-W8 | Phase 5 | 测试覆盖 |
| W9-W10 | Phase 6 | 文档部署 |

**总工期**: 10周
**目标成熟度**: 3/10 → 8/10

---

**创建日期**: 2026-05-14
**预计完成**: 2026-07-21