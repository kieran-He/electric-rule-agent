from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependency import get_conversation_service
from app.schemas.session import (
    SessionCreateRequest,
    SessionCreateResponse,
    TitleGenerateRequest,
    TitleGenerateResponse,
    ExampleQuestion,
)
from app.services.benchmark_service import BenchmarkService
from app.services.conversation_service import ConversationService

router = APIRouter(
    prefix="/session",
    tags=["session"],
    responses={
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"},
    }
)

_benchmark_service = BenchmarkService()


@router.post(
    "/create",
    response_model=SessionCreateResponse,
    summary="Create or Check Session",
    description="Create a new session or check existing one. Returns example questions for new sessions.",
)
def create_session(
    req: SessionCreateRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> SessionCreateResponse:
    session_id, is_new = service.create_session(req.session_id, req.channel)
    
    if is_new:
        questions = _benchmark_service.get_random_questions(count=5)
        example_questions = [
            ExampleQuestion(
                question_id=q.get("question_id"),
                question=q.get("question", ""),
                category=q.get("category"),
            )
            for q in questions
        ]
    else:
        example_questions = []
    
    return SessionCreateResponse(
        session_id=session_id,
        is_new=is_new,
        example_questions=example_questions,
    )


@router.post(
    "/title",
    response_model=TitleGenerateResponse,
    summary="Generate Session Title",
    description="Generate a title for the session based on conversation content.",
)
def generate_title(
    req: TitleGenerateRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> TitleGenerateResponse:
    turn_count = service.get_turn_count(req.session_id)
    
    if turn_count == 0:
        raise HTTPException(status_code=400, detail="无对话内容")
    
    existing_title = service.get_title(req.session_id)
    
    if existing_title:
        return TitleGenerateResponse(
            session_id=req.session_id,
            title=existing_title,
            generated=True,
        )
    
    try:
        title = service.generate_title(req.session_id)
        return TitleGenerateResponse(
            session_id=req.session_id,
            title=title,
            generated=True,
        )
    except Exception as e:
        return TitleGenerateResponse(
            session_id=req.session_id,
            title="对话",
            generated=False,
        )