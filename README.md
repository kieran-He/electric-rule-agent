# 电力政策知识问答 Agent / Power Policy Knowledge QA Agent

## 项目简介 / Overview

**中文**: 本项目是一个基于 RAG 技术的电力政策知识问答系统，支持接入飞书机器人，提供多省份政策文档检索和智能问答服务。系统采用混合检索架构（Vector + BM25 + Reranker），支持 LLM 查询重写和语义扩展，并通过 RAGAS 框架进行效果评估。

**English**: This project is a RAG-based power policy knowledge QA system with Feishu bot integration, supporting multi-province policy document retrieval and intelligent Q&A services. The system uses a hybrid retrieval architecture (Vector + BM25 + Reranker), supports LLM-powered query rewrite and semantic expansion, and evaluates performance through the RAGAS framework.

## 核心特性 / Core Features

### 1. 智能检索 / Intelligent Retrieval

- **Query Rewrite (查询重写)**: LLM 驱动的查询优化，提升检索相关性 / LLM-powered query optimization for improved relevance
- **Query Expansion (查询扩展)**: 语义/同义词扩展，增强召回率 / Semantic/synonym expansion for enhanced recall
- **Hybrid Retrieval (混合检索)**: 向量检索 + BM25 + BGE Reranker 三阶段检索 / Three-stage retrieval: Vector + BM25 + BGE Reranker

### 2. 多省份支持 / Multi-Province Support

- 省份隔离的知识库（每省独立 Chroma collection）/ Province-isolated knowledge bases (separate Chroma collections)
- 省份+全局混合检索 / Province + global hybrid retrieval
- 多省份对比查询 / Multi-province comparison queries
- 自动省份检测（低置信度确认流程）/ Automatic province detection with low-confidence confirmation

### 3. 飞书集成 / Feishu Integration

- 实时消息处理 / Real-time message handling
- Webhook 验证（Token + Signature）/ Webhook verification
- 事件去重 / Event deduplication
- 错误告警推送 / Error alert notifications

### 4. 效果评估 / Evaluation System

- RAGAS 指标（faithfulness, answer_relevancy, context_precision）/ RAGAS metrics
- 批量评估 API / Batch evaluation API
- A/B 对比工作流 / A/B comparison workflow
- 17 项核心指标追踪 / 17 core metrics tracking

## 架构示意 / Architecture

```

用户查询 → Query Rewrite → Query Expansion → Hybrid Retrieval → Reranker → LLM Generation → 答案
    ↓                                                              ↓
飞书 Webhook                                                向量数据库 (Chroma)
                                                                +
                                                            BM25 索引
```

## 快速开始 / Quick Start

### 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

### 配置环境 / Configure Environment

```bash
cp .env.example .env
# 编辑 .env 文件配置必要参数 / Edit .env to configure required parameters
```

### 离线导入文档 / Offline Document Ingestion

```bash
# 省份知识库 / Province KB
python tools/offline_ingest.py --kb-scope province --province-code SN --dedupe true --rebuild false

# 全局知识库 / Global KB
python tools/offline_ingest.py --kb-scope global --dedupe true --rebuild true
```

### 启动服务 / Start Service

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 配置说明 / Configuration

### 核心配置 / Core Configuration

| 变量 / Variable | 说明 / Description | 默认值 / Default |
|----------------|-------------------|-----------------|
| `EMBEDDING_MODEL` | 向量嵌入模型 / Embedding model | `BAAI/bge-small-zh-v1.5` |
| `TOP_K` | 检索文档数量 / Retrieval document count | `8` |
| `CHROMA_PATH` | 向量数据库路径 / Vector DB path | `./data/chroma` |
| `DATABASE_URL` | 数据库连接 / Database connection | `sqlite:///./data/processed/app.db` |

### Query Rewrite 配置 / Query Rewrite Configuration

| 变量 / Variable | 说明 / Description | 默认值 / Default |
|----------------|-------------------|-----------------|
| `QUERY_REWRITE_ENABLED` | 启用 LLM 查询重写 / Enable LLM query rewrite | `true` |
| `QUERY_REWRITE_ALWAYS` | 强制重写所有查询 / Always rewrite all queries | `true` |
| `QUERY_REWRITE_MIN_LENGTH` | 最小重写长度阈值 / Minimum length for rewrite | `10` |
| `QUERY_REWRITE_KEEP_ORIGINAL` | 保留原始查询 / Keep original query | `true` |

### Query Expansion 配置 / Query Expansion Configuration

| 变量 / Variable | 说明 / Description | 默认值 / Default |
|----------------|-------------------|-----------------|
| `QUERY_EXPANSION` | 启用查询扩展 / Enable query expansion | `true` |
| `QUERY_EXPANSION_METHOD` | 扩展方法 (semantic/synonyms) / Expansion method | `semantic` |
| `QUERY_EXPANSION_MAX` | 最大扩展数量 / Maximum expansions | `3` |

### Hybrid Retrieval 配置 / Hybrid Retrieval Configuration

| 变量 / Variable | 说明 / Description | 默认值 / Default |
|----------------|-------------------|-----------------|
| `HYBRID_VECTOR_TOP_K` | 向量检索 Top-K / Vector retrieval Top-K | `15` |
| `HYBRID_BM25_TOP_K` | BM25 检索 Top-K / BM25 retrieval Top-K | `15` |
| `HYBRID_FINAL_TOP_K` | 最终返回文档数 / Final document count | `12` |

### BM25 参数 / BM25 Parameters

| 变量 / Variable | 说明 / Description | 默认值 / Default |
|----------------|-------------------|-----------------|
| `BM25_K1` | BM25 K1 参数（词频饱和度）/ BM25 K1 parameter | `1.5` |
| `BM25_B` | BM25 B 参数（文档长度归一化）/ BM25 B parameter | `0.6` |

### Reranker 配置 / Reranker Configuration

| 变量 / Variable | 说明 / Description | 默认值 / Default |
|----------------|-------------------|-----------------|
| `RERANKER_MODEL` | 重排序模型 / Reranker model | `BAAI/bge-reranker-large` |
| `RERANKER_PRELOAD` | 启动时预加载模型 / Preload model at startup | `true` |
| `RERANKER_MAX_LENGTH` | 最大序列长度 / Maximum sequence length | `512` |
| `RERANK_TOP_K` | 重排序后返回数量 / Post-rerank document count | `8` |

### 飞书配置 / Feishu Configuration

| 变量 / Variable | 说明 / Description | 默认值 / Default |
|----------------|-------------------|-----------------|
| `FEISHU_APP_ID` | 飞书应用 ID / Feishu app ID | - |
| `FEISHU_APP_SECRET` | 飞书应用密钥 / Feishu app secret | - |
| `FEISHU_WEBHOOK_URL` | 告警 Webhook 地址 / Alert webhook URL | - |
| `FEISHU_ALERT_ENABLED` | 启用飞书告警 / Enable Feishu alerts | `false` |
| `FEISHU_MAX_WORKERS` | 最大并发工作线程 / Max concurrent workers | `10` |

### 其他配置 / Other Configuration

| 变量 / Variable | 说明 / Description | 默认值 / Default |
|----------------|-------------------|-----------------|
| `PROVINCE_DEFAULT` | 默认省份代码 / Default province code | `SN` |
| `PROVINCE_CONFIDENCE_THRESHOLD` | 省份检测置信度阈值 / Province detection threshold | `0.7` |
| `CONVERSATION_TTL_MINUTES` | 会话过期时间（分钟）/ Session TTL (minutes) | `120` |
| `LLM_TIMEOUT_SECONDS` | LLM 超时时间（秒）/ LLM timeout (seconds) | `120` |
| `OCR_ENABLED` | 启用 OCR / Enable OCR | `false` |

## API 文档 / API Documentation

### 核心 API / Core APIs

| 端点 / Endpoint | 方法 / Method | 说明 / Description |
|----------------|---------------|-------------------|
| `/admin/health` | GET | 服务健康状态 + 运行模式 / Service health + runtime mode |
| `/query` | POST | 内部查询端点 / Internal query endpoint |
| `/feishu/webhook` | POST | 飞书回调端点 / Feishu callback endpoint |

### 数据导入 API / Ingestion API

| 端点 / Endpoint | 方法 / Method | 说明 / Description |
|----------------|---------------|-------------------|
| `/admin/ingest` | POST | 导入文档到知识库（需启用）/ Ingest docs into KB (requires enable) |

> 注意：当 `INGEST_ENABLED=false` 时，`/admin/ingest` 返回 403。生产环境建议使用离线导入。

### 可观测性 API / Observability APIs

| 端点 / Endpoint | 方法 / Method | 说明 / Description |
|----------------|---------------|-------------------|
| `/metrics` | GET | 实时性能摘要（延迟、Token、查询数、错误）/ Real-time performance summary |
| `/metrics/health` | GET | 指标系统健康状态 / Metrics system health |
| `/metrics/history?hours=24` | GET | 历史性能统计 / Historical performance stats |
| `/metrics/errors?hours=24` | GET | 错误统计摘要 / Error count summary |
| `/metrics/province?hours=24` | GET | 按省份分布的查询统计 / Query distribution by province |
| `/metrics/recent?limit=100` | GET | 最近指标记录 / Recent metrics records |
| `/query/trace/{trace_id}` | GET | 特定查询的详细追踪 / Detailed trace for specific query |

### 评估 API / Evaluation APIs

| 端点 / Endpoint | 方法 / Method | 说明 / Description |
|----------------|---------------|-------------------|
| `/evaluation/recent?limit=50` | GET | 获取最近评估记录（含 RAGAS 分数）/ Get recent evaluations with RAGAS scores |
| `/evaluation/summary?hours=24` | GET | 评估指标汇总（平均 faithfulness 等）/ Evaluation metrics summary |
| `/evaluation/run?batch_size=20` | POST | 手动触发批量评估 / Trigger batch evaluation |
| `/evaluation/pending?limit=50` | GET | 获取待评估的追踪记录 / Get pending traces for evaluation |
| `/metrics/ragas?hours=24` | GET | RAGAS 指标趋势（按小时统计）/ RAGAS metrics trends (hourly) |

## 查询示例 / Query Example

```json
{
  "query": "2026年陕西电力市场中长期交易流程是什么？",
  "session_id": "chat_123:user_456",
  "mode": "auto"
}
```

## 目录结构 / Project Structure

```text
.
├── app/
│   ├── api/                  # API 路由 / API routes
│   │   ├── routes_query.py   # 查询接口 / Query endpoints
│   │   ├── routes_metrics.py # 指标接口 / Metrics endpoints
│   │   └── routes_evaluation.py # 评估接口 / Evaluation endpoints
│   ├── core/                 # 核心模块 / Core modules
│   ├── db/                   # 数据库模型 / Database models
│   ├── langchain/            # LangChain 组件 / LangChain components
│   │   ├── hybrid_retriever.py   # 混合检索器 / Hybrid retriever
│   │   └── orchestrator_hybrid.py # 编排器 / Orchestrator
│   ├── schemas/              # 数据模型 / Data schemas
│   └── services/             # 业务服务 / Business services
├── data/
│   ├── raw/                  # 原始文档 / Raw documents
│   │   ├── global/           # 全国政策 / National policies
│   │   ├── SN/               # 陕西政策 / Shaanxi policies
│   │   └── GD/               # 广东政策 / Guangdong policies
│   ├── chroma/               # 向量数据库 / Vector database
│   └── processed/            # 处理后数据 / Processed data
├── evaluation/               # 评估系统 / Evaluation system
│   ├── run_eval.py           # CLI 入口 / CLI entry point
│   ├── benchmark.json        # 测试问题集 / Test questions
│   └── reports/               # 评估报告 / Evaluation reports
└── tools/
    └── offline_ingest.py     # 离线导入工具 / Offline ingest tool
```

## 文档布局建议 / Recommended Docs Layout

```text
data/docs/
  global/
    ... 全国政策文件 (.pdf/.docx/.txt) / National policy files
  SN/
    ... 陕西政策文件 / Shaanxi policy files
  GD/
    ... 广东政策文件 / Guangdong policy files
```

## 评估系统 / Evaluation System

详细的评估系统文档请参阅 / For detailed evaluation system documentation, see: [evaluation/README.md](evaluation/README.md)

### 快速评估命令 / Quick Evaluation Commands

```bash
# 生成测试问题集 / Generate benchmark questions
python evaluation/run_eval.py generate --docs-path data/docs/SN --output evaluation/benchmark.json --count 100

# 运行评估 / Run evaluation
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --ragas --save-db

# A/B 对比 / A/B comparison
python evaluation/run_eval.py compare eval_20260421_001 eval_20260421_002
```

## 注意事项 / Notes

- `INGEST_ENABLED=false` 表示在线模式仅支持查询，数据刷新使用 `tools/offline_ingest.py`
- `CHROMA_PATH` 是在线读取路径，离线导入输出需保持同一路径
- 导入流程使用健壮的清洗和质量检查，OCR 回退可通过 `enable_ocr` 或 `OCR_ENABLED=true` 启用
- OCR 回退依赖本地 Tesseract 安装和语言包（`chi_sim+eng`）
- Windows OCR 快速配置：
  - `TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe`
  - `TESSDATA_PREFIX=e:/newprojects/firstmodel/tools/tessdata`
- 如果 `GLM_API_KEY` 为空，服务将使用确定性回退响应模板