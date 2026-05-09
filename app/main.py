from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes_admin import router as admin_router
from app.api.routes_evaluation import router as evaluation_router
from app.api.routes_feedback import router as feedback_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_metrics import router as metrics_router
from app.api.routes_query import router as query_router
from app.config import settings
from app.core.exceptions import AppError
from app.core.logger import configure_logging
from app.core.embedding_cache import preload_embedding
from app.langchain.reranker_cache import preload_reranker
from app.services.orchestrator_singleton import preload_orchestrator
from app.db.session import init_db
from app.schemas.error import ErrorResponse
from app.services.session_cleanup import cleanup_task

configure_logging()
init_db()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Preloading models at startup...")
    
    if settings.reranker_preload:
        preload_reranker(settings.reranker_model)
        logger.info(f"Reranker preloaded: {settings.reranker_model}")
    
    preload_embedding(settings.embedding_model)
    logger.info(f"Embedding model preloaded: {settings.embedding_model}")
    
    preload_orchestrator(settings)
    logger.info("Orchestrator preloaded")
    
    cleanup_task.start()
    yield
    cleanup_task.stop()


app = FastAPI(title=settings.app_name, version='3.0.0', lifespan=lifespan)
app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(admin_router)
app.include_router(metrics_router)
app.include_router(evaluation_router)
app.include_router(feedback_router)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    payload = ErrorResponse(error_code=exc.code, message=exc.message, detail=exc.detail)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())
