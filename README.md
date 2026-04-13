# Feishu Power Policy Bot (Multi-Province)

FastAPI service for a Feishu bot that supports:

- Province-isolated policy knowledge bases (each province uses a separate Chroma collection)
- Province + global mixed retrieval
- Multi-province comparison retrieval
- Automatic province detection with low-confidence confirmation flow
- Feishu webhook verification (token + signature) and event deduplication

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
```

3. Start API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## APIs

- `GET /admin/health`: service health.
- `POST /admin/ingest`: ingest docs into `province` or `global` KB.
- `POST /query`: internal query endpoint.
- `POST /feishu/webhook`: Feishu callback endpoint.

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

## Ingest Examples

Province KB (auto resolve path from docs root):

```json
{
  "kb_scope": "province",
  "province_code": "SN",
  "rebuild": true,
  "dedupe": true,
  "cleaning_profile": "robust",
  "enable_ocr": false
}
```

Global KB (auto resolve path from docs root):

```json
{
  "kb_scope": "global",
  "rebuild": true,
  "dedupe": true
}
```

Explicit path override:

```json
{
  "docs_path": "E:/policies/shaanxi",
  "kb_scope": "province",
  "province_code": "SN",
  "rebuild": true,
  "enable_ocr": true
}

Example ingest response fields:

- `resolved_docs_path`
- `files_new`
- `files_updated`
- `files_skipped`
- `ocr_pages_processed`
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

- The ingestion pipeline uses robust cleaning and quality checks. OCR fallback can be enabled by request (`enable_ocr`) or environment (`OCR_ENABLED=true`).
- OCR fallback depends on local Tesseract installation and language pack (`chi_sim+eng`).
- Windows OCR quick config:
  - `TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe`
  - `TESSDATA_PREFIX=e:/newprojects/firstmodel/tools/tessdata` (or your own tessdata directory)
- If `GLM_API_KEY` is empty, the service uses a deterministic fallback response template.
