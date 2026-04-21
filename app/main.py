from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes_admin import router as admin_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.config import settings
from app.core.exceptions import AppError
from app.core.logger import configure_logging
from app.db.session import init_db
from app.schemas.error import ErrorResponse

configure_logging()
init_db()

app = FastAPI(title=settings.app_name, version='3.0.0')
app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(admin_router)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    payload = ErrorResponse(error_code=exc.code, message=exc.message, detail=exc.detail)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'app': settings.app_name}
