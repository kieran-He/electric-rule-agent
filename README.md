# 电力政策知识问答系统

基于 RAG 技术的电力政策知识问答系统，支持飞书机器人集成，提供多省份政策文档检索和智能问答服务。

## 项目简介

本项目采用混合检索架构（Vector + BM25 + BGE Reranker），支持 LLM 查询重写和语义扩展，并通过 RAGAS 框架进行效果评估。

**主要应用场景**：
- 电力市场交易政策咨询
- 多省份政策对比查询
- 企业合规政策解读
- 飞书机器人智能客服

**技术栈**：FastAPI + LangChain + LangGraph + ChromaDB + MiniMax GLM / BGE Embedding

**核心特性**：
- ReAct Agent 多工具协作（政策检索、网络搜索、数据查询）
- WebSocket 实时消息处理（飞书机器人）
- 混合检索 + 智能重排序
- 多轮对话 + 指代消解
- 会话状态持久化（LangGraph Checkpoint）
- 早期终止优化（信息充足检测）
- RAGAS 自动评估

## 系统架构

### 整体架构图

```mermaid
graph TB
    subgraph "用户接入层"
        A1[飞书 WebSocket]
        A2[HTTP API]
    end
    
    subgraph "Agent层"
        B1[ReAct Agent<br/>多轮迭代决策]
        B2[Tool Registry<br/>工具注册中心]
    end
    
    subgraph "工具层"
        C1[retrieve_policy<br/>政策检索]
        C2[web_search<br/>网络搜索]
        C3[fetch_electricity_data<br/>数据查询]
        C4[analyze_statistics<br/>统计分析]
    end
    
    subgraph "检索层"
        D1[Query Rewrite<br/>LLM查询优化]
        D2[Query Expansion<br/>语义扩展]
        D3[Vector检索<br/>ChromaDB]
        D4[BM25检索<br/>关键词匹配]
        D5[Reranker<br/>BGE重排序]
    end
    
    subgraph "数据层"
        E1[data/docs/<br/>原始文档]
        E2[data/chroma/<br/>向量数据库]
        E3[data/cache/<br/>BM25索引]
        E4[SQLite/PostgreSQL<br/>会话&指标]
    end
    
    A1 --> B1
    A2 --> B1
    B1 --> B2
    B2 --> C1
    B2 --> C2
    B2 --> C3
    B2 --> C4
    C1 --> D1
    D1 --> D2
    D2 --> D3
    D2 --> D4
    D3 --> D5
    D4 --> D5
    E1 --> D3
    E2 --> D3
    E3 --> D4
    E4 --> B1
```

### 核心组件说明

| 组件 | 功能 | 模型/技术 |
|------|------|-----------|
| ReAct Agent | 多轮迭代决策，工具调用 | LangGraph + MiniMax |
| Query Rewrite | LLM驱动查询优化 | MiniMax-M2.7 |
| Query Expansion | 语义/同义词扩展 | synonyms.json + semantic |
| Hybrid Retrieval | Vector + BM25 双路召回 | ChromaDB + BM25 |
| Reranker | 精排序 | BAAI/bge-reranker-base |
| LLM Generation | 答案生成 | MiniMax-M2.7 |
| Checkpointer | 会话状态持久化 | SQLite/PostgreSQL |

---

## 接口总览

本系统提供 **23 个接口**：HTTP API 21 个 + 飞书 WebSocket 2 个。

### 接口分类

| 类别 | 数量 | 说明 |
|------|------|------|
| Agent接口 | 1 | ReAct Agent 多轮对话 |
| 查询接口 | 2 | 混合检索查询 |
| 反馈接口 | 3 | 用户反馈管理 |
| 指标接口 | 7 | 性能指标监控 |
| 评估接口 | 4 | RAGAS评估 |
| 导入接口 | 1 | 文档导入 |
| 管理接口 | 3 | 系统管理 |
| 飞书接口 | 2 | WebSocket消息处理 |

---

## HTTP API 接口详细说明

### Agent 接口 (与飞书逻辑完全一致)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/agent/chat` | POST | ReAct Agent对话，支持多轮、工具调用、早期终止 |

**请求示例**：
```json
{
  "query": "山东市场对中长期市场购买电量有什么要求",
  "session_id": "user_123",
  "province_codes": ["SD"],
  "history": ["Q: 陕西规则是什么?", "A: ..."]
}
```

**响应示例**：
```json
{
  "answer": "根据《山东省电力市场交易规则》...",
  "intent": "clause_qa",
  "tool_calls": ["retrieve_policy", "web_search"],
  "citations": [{"doc_name": "山东规则.pdf", "excerpt": "..."}],
  "confidence": 0.85,
  "trace_id": "trace_abc123",
  "detected_provinces": "SD",
  "chart_paths": [],
  "metadata": {"iterations": 2, "elapsed_seconds": 45}
}
```

### 查询接口 (单次混合检索)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/query` | POST | 混合检索查询，不走Agent循环 |
| `/query/trace/{trace_id}` | GET | 获取查询追踪记录 |

### 反馈接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/feedback` | POST | 提交用户反馈（评分、类型、建议） |
| `/feedback/trace/{trace_id}` | GET | 获取某trace的所有反馈 |
| `/feedback/stats` | GET | 获取反馈统计汇总 |

### 指标接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/metrics` | GET | 实时性能摘要 |
| `/metrics/health` | GET | 服务健康状态 |
| `/metrics/history` | GET | 历史性能统计 |
| `/metrics/errors` | GET | 错误类型统计 |
| `/metrics/province` | GET | 省份查询分布 |
| `/metrics/recent` | GET | 最近查询明细 |
| `/metrics/ragas` | GET | RAGAS指标趋势 |

### 评估接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/evaluation/recent` | GET | 最近评估记录 |
| `/evaluation/summary` | GET | 评估指标汇总 |
| `/evaluation/run` | POST | 手动触发批量评估 |
| `/evaluation/pending` | GET | 待评估trace列表 |

### 导入接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/ingest/documents` | POST | 导入文档到向量库 |

### 管理接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/admin/rebuild-index` | POST | 重建向量索引 |
| `/admin/documents` | GET | 获取已导入文档列表 |
| `/admin/health` | GET | 系统健康检查 |

---

## 飞书机器人接口

### 接入方式：WebSocket 长连接

飞书机器人通过 WebSocket 实时接收事件，不使用 HTTP API。

| 事件类型 | 处理函数 | 说明 |
|----------|---------|------|
| `im.message.receive_v1` | `handle_message()` | 消息接收 |
| `card.action.trigger` | `handle_card_action()` | 卡片按钮点击 |

### 飞书处理流程

```
飞书用户消息 → WebSocket → FeishuAgentService
                              ↓
                    检查首次对话 → 推送示例问题卡片
                              ↓
                    回复"正在思考中..."
                              ↓
                    ElectricityAgentGraph.chat()
                              ↓
                    ConversationService 保存会话
                              ↓
                    lark.Client API 回复飞书
```

### 飞书功能特性

| 功能 | 说明 |
|------|------|
| 首次对话提示 | 自动推送示例问题卡片 |
| 问题按钮 | 点击直接发送问题到Agent |
| 刷新换一批 | 重新获取随机示例问题 |
| 会话隔离 | 私聊/群聊独立session_id |
| 多轮对话 | 支持上下文指代消解 |

---

## 二次开发指南

### 1. 添加新工具

在 `app/agent/graph/tools/` 下创建工具文件：

```python
# app/agent/graph/tools/my_tool.py
from langchain_core.tools import tool

@tool
def my_new_tool(query: str, param: str) -> str:
    """工具描述，LLM会根据此描述决定是否调用"""
    # 实现工具逻辑
    return "工具执行结果"
```

在 `tool_registry.py` 中注册：

```python
# app/agent/graph/tools/tool_registry.py
ALL_TOOLS = {
    "retrieve_policy": retrieve_policy,
    "web_search": web_search,
    "my_new_tool": my_new_tool,  # 新增
}
```

### 2. 修改Agent行为

Agent核心逻辑在 `app/agent/graph/electricity_agent_graph.py`：

| 配置项 | 文件位置 | 说明 |
|--------|---------|------|
| 超时时间 | `agent_tool_timeout` | Agent迭代超时限制 |
| 最大迭代 | `agent_max_iterations` | ReAct循环最大次数 |
| 系统提示词 | `react_agent_node.py` | LLM决策指令 |

### 3. 修改检索参数

检索配置在 `app/config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HYBRID_VECTOR_TOP_K` | 20 | 向量检索候选数 |
| `HYBRID_BM25_TOP_K` | 20 | BM25检索候选数 |
| `RERANK_TOP_K` | 12 | 重排序返回数 |
| `QUERY_REWRITE_ENABLED` | true | 启用查询重写 |
| `QUERY_EXPANSION_MAX` | 3 | 最大查询扩展数 |

### 4. 添加新省份

```bash
# 1. 创建省份目录
mkdir data/docs/GD

# 2. 放入政策文档
# data/docs/GD/*.pdf

# 3. 离线导入
python tools/offline_ingest.py --kb-scope province --province-code GD

# 4. 更新配置
# app/config.py: PROVINCE_DEFAULTS = ["SN", "SD", "GD"]
```

### 5. 自定义飞书卡片

飞书卡片构建在 `app/utils/feishu_card_builder.py`：

```python
class FeishuCardBuilder:
    def build_custom_card(self, data: dict) -> dict:
        """构建自定义卡片"""
        return {
            "schema": "2.0",
            "header": {...},
            "body": {"elements": [...]}
        }
```

### 6. 扩展评估指标

评估逻辑在 `evaluation/ragas_evaluator.py`，可添加自定义指标：

```python
# 添加新评估维度
custom_metrics = {
    "policy_coverage": PolicyCoverageMetric(),
    "citation_accuracy": CitationAccuracyMetric(),
}
```

---

## 配置参数速查

### Agent配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `AGENT_TOOL_TIMEOUT` | 60 | Agent迭代超时(秒) |
| `AGENT_MAX_ITERATIONS` | 5 | 最大迭代次数 |
| `TOOLS_ENABLED` | retrieve_policy,web_search,... | 启用的工具列表 |

### 检索配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODEL` | BAAI/bge-small-zh-v1.5 | 向量嵌入模型 |
| `RERANKER_MODEL` | BAAI/bge-reranker-base | 重排序模型 |
| `HYBRID_VECTOR_TOP_K` | 20 | 向量检索候选数 |
| `HYBRID_BM25_TOP_K` | 20 | BM25检索候选数 |
| `RERANK_TOP_K` | 12 | 重排序返回数 |
| `RRF_K` | 60 | RRF融合参数 |

### LLM配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL` | MiniMax模型 | 生成模型 |
| `LLM_TIMEOUT_SECONDS` | 120 | LLM调用超时 |
| `QUERY_REWRITE_ENABLED` | true | 查询重写 |
| `QUERY_EXPANSION_METHOD` | semantic | 扩展方法 |

### 飞书配置

| 参数 | 说明 |
|------|------|
| `FEISHU_APP_ID` | 飞书应用ID |
| `FEISHU_APP_SECRET` | 飞书应用密钥 |
| `FEISHU_MAX_WORKERS` | 并发处理线程数 |

---

## 常见问题

**Q: Agent超时无答案？**
A: 检查 `AGENT_TOOL_TIMEOUT` 配置，增大超时时间或优化检索效率。

**Q: 飞书消息无响应？**
A: 检查 `FEISHU_APP_ID/SECRET` 配置，确认WebSocket连接成功。

**Q: 如何跳过Agent直接检索？**
A: 使用 `/query` 接口而非 `/agent/chat`。

**Q: 如何禁用web_search？**
A: 设置 `TOOLS_ENABLED=retrieve_policy` 或 `WEB_SEARCH_ENABLED=false`。

**Q: 如何添加自定义停词/同义词？**
A: 编辑 `data/dict/stopwords.txt` 和 `data/dict/synonyms.json`。

---

## 目录结构

```text
firstmodel/
├── app/                        # 核心应用
│   ├── api/                    # API路由 (21个接口)
│   │   ├── routes_agent.py     # Agent接口
│   │   ├── routes_query.py     # 查询接口
│   │   ├── routes_feedback.py  # 反馈接口
│   │   ├── routes_metrics.py   # 指标接口
│   │   ├── routes_evaluation.py # 评估接口
│   │   ├── routes_ingest.py    # 导入接口
│   │   └── routes_admin.py     # 管理接口
│   ├── agent/                  # Agent模块
│   │   ├── graph/              # LangGraph图定义
│   │   │   ├── electricity_agent_graph.py  # Agent主图
│   │   │   ├── nodes/          # 节点实现
│   │   │   │   ├── react_agent_node.py     # ReAct决策
│   │   │   │   └── tool_executor_node.py   # 工具执行
│   │   │   ├── tools/          # 工具定义
│   │   │   └── checkpointer/   # 会话持久化
│   │   └── agent_singleton.py  # Agent单例
│   ├── core/                   # 核心模块
│   ├── db/                     # 数据库模型
│   ├── langchain/              # LangChain组件
│   │   ├── hybrid_retriever.py # 混合检索器
│   │   ├── query_rewriter.py   # 查询重写
│   │   └ query_expander.py     # 查询扩展
│   │   └ orchestrator_hybrid.py # 编排器
│   ├── schemas/                # 数据模型
│   ├── services/               # 业务服务
│   │   ├── feishu_agent_service.py  # 飞书服务
│   │   ├── benchmark_service.py     # 示例问题服务
│   │   ├── conversation_service.py  # 会话管理
│   │   └ orchestrator_singleton.py  # 检索编排单例
│   └── utils/                  # 工具函数
│       ├── feishu_card_builder.py   # 飞书卡片构建
│       └── markdown_to_feishu.py     # Markdown转换
├── evaluation/                 # 评估系统
│   ├── benchmark.json          # 测试基准问题(100条)
│   └ run_eval.py               # CLI评估入口
│   └ ragas_evaluator.py        # RAGAS评估器
├── dataprocess/                # 数据处理管道
├── data/                       # 数据存储
│   ├── docs/                   # 原始文档(按省份)
│   ├── chroma/                 # 向量数据库
│   ├── cache/                  # BM25索引缓存
│   └ dict/                     # 字典数据
│   └ processed/                # 处理后数据
├── tests/                      # 测试文件
├── tools/                      # 工具脚本
├── feishu_bot.py               # 飞书机器人入口
├── main.py                     # HTTP API入口
├── config.py                   # 配置管理
└── README.md                   # 本文档
```

---

## 启动服务

### HTTP API 服务

```bash
# 开发模式
uvicorn app.main:app --reload --port 8000

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# API文档
# http://localhost:8000/docs
```

### 飞书机器人服务

```bash
# WebSocket模式
python -m app.feishu_bot
```

---

## 更多文档

- [评估系统说明](evaluation/README.md)
- [数据处理流程](dataprocess/README.md)
- [API接口文档](http://localhost:8000/docs)

### 处理流程图

```mermaid
sequenceDiagram
    participant U as 用户
    participant Q as Query处理
    participant R as 检索层
    participant G as LLM生成
    
    U->>Q: 发送查询
    Q->>Q: Query Rewrite (LLM优化)
    Q->>Q: Query Expansion (语义扩展)
    Q->>Q: 省份检测
    
    Q->>R: 执行检索
    R->>R: Vector检索 (Top-K1)
    R->>R: BM25检索 (Top-K2)
    R->>R: 合并候选集
    R->>R: Reranker重排序
    R->>G: 返回 Top-K 文档
    
    G->>G: 生成答案
    G->>U: 返回结果
```

### 核心组件说明

| 组件 | 功能 | 模型/技术 |
|------|------|-----------|
| Query Rewrite | LLM驱动查询优化，提升检索相关性 | MINIMAX-M2.7 |
| Query Expansion | 语义/同义词扩展，增强召回率 | synonyms.json + semantic |
| Hybrid Retrieval | Vector + BM25 双路召回 | ChromaDB + BM25 |
| Reranker | 精排序，筛选最相关文档 | BAAI/bge-reranker-large |
| LLM Generation | 答案生成 | MINIMAX-M2.7 |

## 快速开始

### 环境准备

```bash
# 克隆项目
git clone <repo_url>
cd firstmodel

# 创建虚拟环境（可选）
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 配置说明

```bash
# 复制环境配置模板
cp .env.example .env

# 编辑 .env 配置必要参数
# 主要配置项：
# - GLM_API_KEY: 智谱AI API密钥（必需）
# - EMBEDDING_MODEL: 向量嵌入模型路径
# - CHROMA_PATH: 向量数据库路径
```

### 启动服务

```bash
# 开发模式（带热重载）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 核心功能

### 智能检索

- **Query Rewrite**: LLM驱动的查询优化，支持保留原始查询、最小长度阈值控制
- **Query Expansion**: 语义扩展、同义词扩展，最大扩展数量可配置
- **Hybrid Retrieval**: 向量检索 + BM25 + BGE Reranker 三阶段检索，可配置各阶段 Top-K

### 多省份支持

- 省份隔离的知识库（每省独立 Chroma collection）
- 省份+全局混合检索模式
- 自动省份检测（置信度阈值可配置）
- 低置信度时触发确认流程

### 飞书集成

- 实时消息处理
- Webhook 验证（Token + Signature）
- 事件去重
- 错误告警推送

### 效果评估

- RAGAS 指标（faithfulness, answer_relevancy, context_precision）
- 批量评估 API
- A/B 对比工作流
- 17项核心指标追踪

## 目录结构

```text
firstmodel/
├── app/                        # 核心应用
│   ├── api/                    # API路由
│   │   ├── routes_query.py     # 查询接口
│   │   ├── routes_metrics.py   # 指标接口
│   │   └── routes_evaluation.py # 评估接口
│   ├── core/                   # 核心模块
│   ├── db/                     # 数据库模型
│   ├── langchain/              # LangChain组件
│   │   ├── hybrid_retriever.py # 混合检索器
│   │   └── orchestrator_hybrid.py # 编排器
│   ├── schemas/                # 数据模型
│   ├── services/               # 业务服务
│   └── utils/                  # 工具函数
├── dataprocess/                # 数据处理管道
│   ├── pipeline.py             # 文档处理主流程
│   ├── parsers/                # PDF/DOCX解析器
│   └── chunkers/               # LLM智能分块
├── evaluation/                 # 评估系统
│   ├── run_eval.py             # CLI评估入口
│   ├── benchmark.json          # 测试基准问题
│   ├── experiments/            # 实验数据
│   └── reports/                # 评估报告
├── tests/                      # 测试文件
├── tools/                      # 工具脚本
│   ├── offline_ingest.py       # 离线导入工具
│   ├── smoke_rag.py            # 快速测试工具
│   └── tessdata/               # OCR语言数据
├── scripts/                    # 辅助脚本
├── docs/                       # 文档
├── data/                       # 数据存储
│   ├── docs/                   # 原始政策文档（按省份）
│   │   ├── global/             # 全国政策
│   │   ├── SN/                 # 陕西
│   │   ├── SX/                 # 山西
│   │   └── GS/                 # 甘肃
│   ├── chroma/                 # 向量数据库
│   ├── cache/                  # BM25索引缓存
│   ├── dict/                   # 字典数据
│   │   ├── stopwords.txt       # 停词表
│   │   ├── synonyms.json       # 同义词
│   │   └── power_policy.txt    # 领域词典
│   └── processed/              # 处理后数据
│       └── SN/_manifest.json   # 处理记录
├── .env.example                # 环境配置模板
├── .gitignore                  # Git忽略规则
├── README.md                   # 项目说明
└── requirements.txt            # 依赖列表
```

## API接口

### 查询接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/query` | POST | 内部查询端点 |
| `/feishu/webhook` | POST | 飞书回调端点 |

### 管理接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/admin/health` | GET | 服务健康状态 |
| `/admin/ingest` | POST | 导入文档（需启用） |

### 指标接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/metrics` | GET | 实时性能摘要 |
| `/metrics/history` | GET | 历史性能统计 |
| `/metrics/errors` | GET | 错误统计 |
| `/metrics/province` | GET | 省份查询分布 |
| `/query/trace/{trace_id}` | GET | 查询详细追踪 |

### 评估接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/evaluation/recent` | GET | 最近评估记录 |
| `/evaluation/summary` | GET | 评估指标汇总 |
| `/evaluation/run` | POST | 手动触发批量评估 |
| `/metrics/ragas` | GET | RAGAS指标趋势 |

## 开发指南

### 本地开发

```bash
# 启动开发服务
uvicorn app.main:app --reload

# 查看API文档
# http://localhost:8000/docs
# http://localhost:8000/redoc
```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_retriever.py -v

# 运行覆盖率测试
pytest tests/ --cov=app
```

### 添加新省份

1. 在 `data/docs/` 下创建省份目录（如 `GD/`）
2. 放入政策文档（PDF/DOCX）
3. 运行离线导入：
   ```bash
   python tools/offline_ingest.py --kb-scope province --province-code GD --dedupe true
   ```
4. 更新 `app/config.py` 中的 `PROVINCE_DEFAULT`

### 调试技巧

```bash
# 快速验证RAG功能
python tools/smoke_rag.py --query "陕西电力交易规则"

# 查看向量数据库状态
python -c "from app.core.vector_store import get_collection_info; print(get_collection_info('SN'))"
```

## 部署说明

### 生产环境配置

```env
# 必需配置
GLM_API_KEY=<your_api_key>
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
CHROMA_PATH=./data/chroma

# 性能配置
TOP_K=8
HYBRID_VECTOR_TOP_K=15
HYBRID_BM25_TOP_K=15
RERANK_TOP_K=8

# 安全配置
INGEST_ENABLED=false
FEISHU_ALERT_ENABLED=true
```

### 离线数据导入

```bash
# 导入省份知识库
python tools/offline_ingest.py --kb-scope province --province-code SN --dedupe true

# 导入全局知识库
python tools/offline_ingest.py --kb-scope global --dedupe true --rebuild true

# 查看导入状态
python tools/offline_ingest.py --status
```

### 服务启动

```bash
# 生产环境启动
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker启动（可选）
docker build -t power-policy-qa .
docker run -p 8000:8000 power-policy-qa
```

### 性能调优

- 调整 `HYBRID_VECTOR_TOP_K` 和 `HYBRID_BM25_TOP_K` 平衡召回率与延迟
- 启用 `RERANKER_PRELOAD=true` 减少首次请求延迟
- 配置 `FEISHU_MAX_WORKERS` 控制飞书并发处理数

## 数据处理流程

### 文档导入流程

```mermaid
flowchart LR
    A[原始文档<br/>PDF/DOCX] --> B[解析器<br/>pdfplumber/ocr]
    B --> C[文本清洗<br/>去噪/格式化]
    C --> D[LLM分块<br/>智能切分]
    D --> E[元数据提取<br/>省份/类型/日期]
    E --> F[向量嵌入<br/>BGE Embedding]
    F --> G[ChromaDB<br/>向量存储]
    E --> H[BM25索引<br/>关键词索引]
```

### 向量索引构建

```bash
# dataprocess模块处理流程
python -m dataprocess.pipeline --input data/docs/SN --output data/processed/SN

# 处理步骤：
# 1. 文档解析 -> 提取文本
# 2. LLM分块 -> 智能切分
# 3. 元数据提取 -> 省份/类型/来源
# 4. 向量嵌入 -> BGE Embedding
# 5. Chroma存储 -> 向量索引
```

### BM25索引构建

BM25索引在离线导入时自动构建，存储在 `data/cache/` 目录：
- `bm25_sn.pkl` - 陕西BM25索引
- `bm25_global.pkl` - 全局BM25索引

## 评估系统

详细文档请参阅 [evaluation/README.md](evaluation/README.md)

### 快速评估命令

```bash
# 生成测试问题集
python evaluation/run_eval.py generate --docs-path data/docs/SN --output evaluation/benchmark.json --count 100

# 运行评估
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --ragas --save-db

# A/B对比
python evaluation/run_eval.py compare eval_001 eval_002
```

### RAGAS指标说明

| 指标 | 说明 | 目标值 |
|------|------|--------|
| faithfulness | 答案与上下文一致性 | > 0.85 |
| answer_relevancy | 答案与问题相关性 | > 0.80 |
| context_precision | 检索上下文精确度 | > 0.75 |

## 配置参数速查

### 核心配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODEL` | BAAI/bge-small-zh-v1.5 | 向量嵌入模型 |
| `TOP_K` | 8 | 最终返回文档数 |
| `CHROMA_PATH` | ./data/chroma | 向量数据库路径 |

### 检索配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HYBRID_VECTOR_TOP_K` | 15 | 向量检索候选数 |
| `HYBRID_BM25_TOP_K` | 15 | BM25检索候选数 |
| `BM25_K1` | 1.5 | BM25词频饱和度 |
| `BM25_B` | 0.6 | BM25长度归一化 |
| `RERANK_TOP_K` | 8 | 重排序返回数 |

### Query处理配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QUERY_REWRITE_ENABLED` | true | 启用查询重写 |
| `QUERY_EXPANSION` | true | 启用查询扩展 |
| `QUERY_EXPANSION_METHOD` | semantic | 扩展方法 |
| `QUERY_EXPANSION_MAX` | 3 | 最大扩展数 |

## 常见问题

**Q: 服务启动后查询返回空结果？**
A: 检查 `data/chroma/` 目录是否存在向量数据，运行离线导入：
```bash
python tools/offline_ingest.py --status
```

**Q: 如何添加新的政策文档？**
A: 将文档放入对应省份目录，运行离线导入：
```bash
python tools/offline_ingest.py --kb-scope province --province-code SN
```

**Q: 飞书机器人无法接收消息？**
A: 检查飞书配置（`FEISHU_APP_ID`, `FEISHU_APP_SECRET`）和Webhook验证。

**Q: OCR功能如何启用？**
A: 安装Tesseract并配置：
```env
OCR_ENABLED=true
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
TESSDATA_PREFIX=./tools/tessdata
```