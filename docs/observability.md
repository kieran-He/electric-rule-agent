# Observability Guide

可观测性系统包含三大模块：Metrics、Trace、Logging。

## 1. MetricsStore (性能监控)

### 1.1 实时指标

`MetricsStore` 收集以下实时指标（内存滑动窗口，100样本）：

| 指标类型 | 方法 | 说明 |
|----------|------|------|
| 延迟 | `record_latency(ms, category)` | retrieval/llm/total |
| Token | `record_tokens(input, output)` | LLM token 消耗 |
| 查询 | `record_query(province_code)` | 省份分布统计 |
| 错误 | `record_error(error_type)` | 错误类型计数 |

### 1.2 历史数据持久化

每次查询执行后，数据自动持久化到 `metrics_record` 表：

```python
# app/langchain/orchestrator_hybrid.py
metrics_store.save_to_db(
    db=self.db,
    trace_id=trace_id,
    session_id=req.session_id,
    request_id=getattr(req, 'request_id', None),
    user_id=getattr(req, 'user_id', None),
    retrieval_latency_ms=retrieval_latency,
    llm_latency_ms=llm_latency,
    total_latency_ms=total_latency,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    province_code=province_code,
    success=True,
)
```

### 1.3 API 端点

| 端点 | 说明 | 示例响应 |
|------|------|----------|
| `/metrics` | 实时汇总 | `{"latency_avg_ms": 150, "tokens_avg": 150}` |
| `/metrics/history?hours=24` | 历史统计 | `{"count": 100, "total_tokens": 15000}` |
| `/metrics/errors?hours=24` | 错误统计 | `{"error_count": 5}` |
| `/metrics/province?hours=24` | 省份分布 | `{"SN": 50, "GD": 30}` |
| `/metrics/recent?limit=100` | 最近记录 | `[{"trace_id": "...", ...}]` |

## 2. Trace (请求追踪)

### 2.1 TraceRecord 字段

每次查询生成唯一 `trace_id`，记录以下信息：

| 字段 | 类型 | 说明 |
|------|------|------|
| trace_id | String | 唯一请求标识 |
| session_id | String | 会话标识 |
| raw_query | Text | 用户原始查询 |
| rewritten_query | Text | 查询改写结果 |
| retrieved_doc_ids | Text (JSON) | 检索文档ID列表 |
| rerank_scores | Text (JSON) | 重排序分数 |
| input_tokens | Integer | 输入Token数 |
| output_tokens | Integer | 输出Token数 |
| retrieval_latency_ms | Integer | 检索延迟 |
| llm_latency_ms | Integer | LLM延迟 |
| success | Boolean | 是否成功 |
| error_type | String | 错误类型 |

### 2.2 Trace API

```bash
curl http://localhost:8000/query/trace/trace_abc123
```

响应示例：

```json
{
  "trace_id": "trace_abc123",
  "session_id": "session_xyz",
  "raw_query": "2026年陕西电力交易流程",
  "latency_ms": 350,
  "input_tokens": 120,
  "output_tokens": 80,
  "retrieved_doc_ids": [1, 5, 8],
  "rerank_scores": [0.85, 0.72, 0.65],
  "success": true
}
```

## 3. Structured Logging

### 3.1 日志格式

结构化日志输出到 `data/processed/app_structured.json`：

```json
{
  "timestamp": "2026-04-27T03:48:10Z",
  "level": "INFO",
  "logger": "app.langchain.orchestrator_hybrid",
  "message": "Query completed",
  "trace_id": "trace_abc123",
  "session_id": "session_xyz",
  "request_id": "req_001",
  "user_id": "user_001",
  "module": "orchestrator_hybrid",
  "function": "run",
  "line": 260
}
```

### 3.2 上下文变量

通过 `logging_context.py` 设置追踪字段：

```python
from app.core.logging_context import set_trace_id, set_session_id, set_request_id, set_user_id

set_trace_id("trace_abc123")
set_session_id("session_xyz")
set_request_id("req_001")
set_user_id("user_001")
```

## 4. Feishu Alerting

### 4.1 配置

启用飞书群错误报警：

```bash
# .env
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx
FEISHU_ALERT_ENABLED=true
```

### 4.2 报警消息格式

当 ERROR 级日志触发时，自动推送：

```
⚠️ [ERROR] app.langchain.llm
LLM invoke failed: Timeout after 30s
trace_id: trace_abc123
session_id: session_xyz
request_id: req_001
user_id: user_001
```

### 4.3 实现

`FeishuAlertHandler` 自动捕获所有 ERROR 级日志并推送：

```python
# app/core/logger.py
if settings.feishu_alert_enabled and settings.feishu_webhook_url:
    feishu_handler = create_feishu_handler(settings.feishu_webhook_url)
    root.addHandler(feishu_handler)
```

## 5. Health Check

### 5.1 组件检查

`/admin/health` 检查 5 个核心组件：

| 组件 | 检查内容 |
|------|----------|
| Database | SQLite 连接状态 |
| ChromaDB | Collection 存在性 |
| LLM | API Key 配置 |
| Reranker | 模型加载状态 |
| BM25 | 索引构建状态 |

### 5.2 示例响应

```json
{
  "overall": "ok",
  "components": [
    {"name": "database", "status": "ok"},
    {"name": "chroma", "status": "ok", "collections": 2},
    {"name": "llm", "status": "ok", "model": "glm-4"},
    {"name": "reranker", "status": "ok", "loaded": true},
    {"name": "bm25", "status": "ok", "docs": 150}
  ]
}
```

## 6. TTL Cleanup

### 6.1 自动清理

会话和 Trace 数据自动清理（TTL=120分钟）：

```python
# app/services/session_cleanup.py
cleanup_service.run_cleanup(
    ttl_minutes=settings.conversation_ttl_minutes
)
```

清理内容：
- `ConversationState` 表
- `ConversationTurn` 表
- `TraceRecord` 表
- `MetricsRecord` 表

## 7. 使用示例

### 7.1 查询实时指标

```bash
curl http://localhost:8000/metrics
```

### 7.2 查询历史Token消耗

```bash
curl http://localhost:8000/metrics/history?hours=24
```

### 7.3 追踪特定请求

```bash
# 先获取 trace_id（从 QueryAnswer.trace_id 字段）
curl http://localhost:8000/query/trace/trace_abc123
```

### 7.4 分析省份分布

```bash
curl http://localhost:8000/metrics/province?hours=24
```

## 8. 数据库表结构

### 8.1 metrics_record

| 字段 | 类型 | 索引 |
|------|------|------|
| id | Integer (PK) | - |
| trace_id | String(128) | ✓ |
| session_id | String(128) | ✓ |
| request_id | String(128) | ✓ |
| user_id | String(128) | ✓ |
| retrieval_latency_ms | Integer | - |
| llm_latency_ms | Integer | - |
| total_latency_ms | Integer | - |
| input_tokens | Integer | - |
| output_tokens | Integer | - |
| province_code | String(16) | - |
| error_type | String(64) | - |
| error_message | Text | - |
| success | Boolean | - |
| created_at | DateTime | ✓ |

### 8.2 trace_record

与 metrics_record 类似，额外包含：
- raw_query, rewritten_query (查询内容)
- retrieved_doc_ids, rerank_scores (检索详情)
- intent (意图分类)