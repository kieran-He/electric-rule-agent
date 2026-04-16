from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request

from app.config import settings
from app.feishu import FeishuClient
from app.generator import GLMClient, LLMGenerationError
from app.ingest_path import build_missing_path_hint, resolve_docs_path
from app.ingestion import DocumentIngestor
from app.province import ProvinceDetector
from app.repository import ChromaPolicyRepository, RepositoryError
from app.schemas import HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.security import EventDeduplicator, verify_signature, verify_token
from app.service import PolicyQueryService, QueryPlanner
from app.session import SessionStore


app = FastAPI(title=settings.app_name)
repository = ChromaPolicyRepository(
    persist_directory=settings.chroma_path, embedding_model_name=settings.embedding_model
)
generator = GLMClient(
    api_key=settings.glm_api_key,
    endpoint=settings.glm_endpoint,
    model=settings.glm_model,
    timeout_seconds=settings.glm_timeout_seconds,
)
service = PolicyQueryService(
    repository=repository,
    generator=generator,
    detector=ProvinceDetector(),
    sessions=SessionStore(max_turns=6),
    planner=QueryPlanner(),
)
ingestor = DocumentIngestor(repository, index_path=settings.ingest_index_path)
feishu_client = FeishuClient(settings.feishu_app_id, settings.feishu_app_secret)
dedup = EventDeduplicator(ttl_seconds=settings.event_ttl_seconds)


def format_response_text(resp: QueryResponse) -> str:
    lines = [f"结论：{resp.conclusion}"]
    if resp.provincial_evidence:
        lines.append("省级依据：")
        for c in resp.provincial_evidence[:3]:
            lines.append(f"- [{c.province_code or 'NA'}] {c.source_name} {c.snippet}")
    if resp.global_evidence:
        lines.append("通用依据：")
        for c in resp.global_evidence[:2]:
            lines.append(f"- {c.source_name} {c.snippet}")
    if resp.differences:
        lines.append(f"差异说明：{resp.differences}")
    lines.append(f"建议追问：{resp.follow_up}")
    return "\n".join(lines)


@app.get("/admin/health", response_model=HealthResponse)
def health() -> HealthResponse:
    mode_state = (
        f"ingest_enabled={settings.ingest_enabled}, chroma_path={settings.chroma_path}"
    )
    llm_state = (
        f"llm strict mode=enabled, ready=true, model={settings.glm_model}"
        if generator.ready
        else "llm strict mode=enabled, ready=false (GLM_API_KEY is empty; /query will return 503)"
    )
    retrieval_state = f"retrieval embedder={repository.embedder_name}"
    if repository.ready:
        return HealthResponse(
            status="ok",
            vector_store_ready=True,
            glm_ready=generator.ready,
            message=f"service is ready; {mode_state}; {retrieval_state}; {llm_state}",
        )
    return HealthResponse(
        status="degraded",
        vector_store_ready=False,
        glm_ready=generator.ready,
        message=f"vector store init failed: {repository.init_error}; {mode_state}; {retrieval_state}; {llm_state}",
    )


@app.post("/admin/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    if not settings.ingest_enabled:
        raise HTTPException(
            status_code=403,
            detail="ingest endpoint is disabled in online mode; please run offline ingestion script",
        )
    if req.kb_scope.value == "province" and not req.province_code:
        raise HTTPException(status_code=400, detail="province_code is required when kb_scope=province")
    if req.cleaning_profile != "robust":
        raise HTTPException(status_code=400, detail="only cleaning_profile=robust is supported currently")
    if req.chunk_overlap >= req.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size")
    resolved_docs_path = resolve_docs_path(
        kb_scope=req.kb_scope.value,
        province_code=req.province_code,
        docs_path=req.docs_path,
        docs_root=req.docs_root,
        default_docs_root=settings.docs_root,
    )
    try:
        stats = ingestor.ingest_path(
            docs_path=resolved_docs_path,
            kb_scope=req.kb_scope.value,
            province_code=req.province_code,
            rebuild=req.rebuild,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            enable_ocr=settings.ocr_enabled if req.enable_ocr is None else req.enable_ocr,
            dedupe=req.dedupe,
            min_ch_ratio=settings.ocr_min_ch_ratio,
            max_replacement_ratio=settings.ocr_max_replacement_ratio,
            empty_page_threshold=settings.ocr_empty_page_threshold,
        )
        return IngestResponse(
            success=True,
            files_processed=stats.files_processed,
            chunks_created=stats.chunks_created,
            kb_scope=req.kb_scope,
            province_code=req.province_code,
            resolved_docs_path=resolved_docs_path,
            files_new=stats.files_new,
            files_updated=stats.files_updated,
            files_skipped=stats.files_skipped,
            ocr_pages_processed=stats.ocr_pages_processed,
            message="ingestion completed",
        )
    except FileNotFoundError as exc:
        hint = build_missing_path_hint(
            path=resolved_docs_path,
            kb_scope=req.kb_scope.value,
            province_code=req.province_code,
        )
        raise HTTPException(status_code=400, detail=hint) from exc
    except (RepositoryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        return service.process(req)
    except LLMGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RepositoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/feishu/webhook")
async def feishu_webhook(
    request: Request,
    x_lark_request_timestamp: Optional[str] = Header(default=None),
    x_lark_signature: Optional[str] = Header(default=None),
):
    body_bytes = await request.body()
    payload = await request.json()

    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    if not verify_token(payload, settings.feishu_token):
        raise HTTPException(status_code=401, detail="invalid verification token")

    if not verify_signature(
        x_lark_request_timestamp or "",
        body_bytes,
        x_lark_signature or "",
        settings.feishu_signing_secret,
    ):
        raise HTTPException(status_code=401, detail="invalid signature")

    event_id = payload.get("header", {}).get("event_id", "")
    if event_id and dedup.seen(event_id):
        return {"ok": True, "message": "duplicate event ignored"}

    event = payload.get("event", {})
    message = event.get("message", {})
    sender = event.get("sender", {})
    text = message.get("content", "")
    chat_id = message.get("chat_id")
    user_id = sender.get("sender_id", {}).get("open_id") or "unknown"
    if text.startswith("{"):
        import json

        try:
            text = json.loads(text).get("text", "")
        except json.JSONDecodeError:
            pass

    session_id = f"{chat_id}:{user_id}"
    result = service.process(QueryRequest(query=text, session_id=session_id))
    out_text = format_response_text(result)
    if chat_id:
        feishu_client.send_text(chat_id=chat_id, text=out_text)
    return {"ok": True}
