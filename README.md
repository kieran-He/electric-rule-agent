# Feishu Power Policy Bot (Multi-Province)

FastAPI service for a Feishu bot that supports:

- Province-isolated policy knowledge bases (each province uses a separate Chroma collection)
- Province + global mixed retrieval
- Multi-province comparison retrieval
- Automatic province detection with low-confidence confirmation flow
- Feishu webhook verification (token + signature) and event deduplication
- Offline ingestion + online query-only operation mode

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
```

3. Offline ingest docs (recommended for production):

```bash
python tools/offline_ingest.py --kb-scope province --province-code SN --dedupe true --rebuild false
```

4. Start API (query-only by default):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## APIs

- `GET /admin/health`: service health + runtime mode.
- `POST /admin/ingest`: ingest docs into KB (disabled by default; returns 403 when `INGEST_ENABLED=false`).
- `POST /query`: internal query endpoint.
- `POST /feishu/webhook`: Feishu callback endpoint.

### Observability APIs

- `GET /metrics`: real-time performance summary (latency, tokens, query counts, errors).
- `GET /metrics/health`: metrics system health status.
- `GET /metrics/history?hours=24`: historical performance stats (avg latency, total tokens).
- `GET /metrics/errors?hours=24`: error count summary.
- `GET /metrics/province?hours=24`: query distribution by province.
- `GET /metrics/recent?limit=100`: recent metrics records.
- `GET /query/trace/{trace_id}`: detailed trace for a specific query.

### Feishu Alert Configuration

Enable real-time error alerts to Feishu group:

```bash
# Set environment variables
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx
FEISHU_ALERT_ENABLED=true
```

When ERROR level logs occur, alerts will be sent to Feishu with trace_id, session_id, request_id context.

## Recommended Docs Layout

```text
data/docs/
  global/
    ... national policy files (.pdf/.docx/.txt)
  SN/
    ... shaanxi policy files
  GD/
    ... guangdong policy files
```

## Offline Ingest Examples

Province KB (auto resolve path from docs root):

```bash
python tools/offline_ingest.py --kb-scope province --province-code SN --dedupe true --rebuild true
```

Global KB (auto resolve path from docs root):

```bash
python tools/offline_ingest.py --kb-scope global --dedupe true --rebuild true
```

Explicit path override:

```bash
python tools/offline_ingest.py --kb-scope province --province-code SN --docs-path E:/policies/shaanxi --rebuild true --enable-ocr true
```

## Query Example

```json
{
  "query": "2026年陕西电力市场中长期交易流程是什么？",
  "session_id": "chat_123:user_456",
  "mode": "auto"
}
```

## Notes

- `INGEST_ENABLED=false` means online mode is query-only; use `tools/offline_ingest.py` for data refresh.
- `CHROMA_PATH` is the online read path. Keep offline ingest output in the same path (or switch via snapshot promotion).
- The ingestion pipeline uses robust cleaning and quality checks. OCR fallback can be enabled by request (`enable_ocr`) or environment (`OCR_ENABLED=true`).
- OCR fallback depends on local Tesseract installation and language pack (`chi_sim+eng`).
- Windows OCR quick config:
  - `TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe`
  - `TESSDATA_PREFIX=e:/newprojects/firstmodel/tools/tessdata` (or your own tessdata directory)
- If `GLM_API_KEY` is empty, the service uses a deterministic fallback response template.
