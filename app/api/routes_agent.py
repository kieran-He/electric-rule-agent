from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agent.agent_singleton import agent_singleton
from app.core.dependency import get_dialog_manager, get_benchmark_service, get_title_generator, get_conversation_service
from app.core.exceptions import AppError
from app.core.logging_context import set_trace_id, set_session_id
from app.db.session import SessionLocal
from app.schemas.agent import AgentRequest, AgentResponse, BenchmarkResponse, TitleRequest, TitleResponse
from app.services.benchmark_service import BenchmarkService
from app.services.conversation_service import ConversationService
from app.services.title_generator import TitleGenerator


router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    responses={
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"},
    }
)


def get_db():
    """Get database session for dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post(
    "/chat",
    response_model=AgentResponse,
    summary="Agent Chat",
    description="Process user query through PowerPolicyAgent with multi-turn conversation support.",
    responses={
        200: {
            "description": "Successful agent response",
            "content": {
                "application/json": {
                    "example": {
                        "answer": "根据《陕西省电力市场交易实施细则》...",
                        "intent": "clause_qa",
                        "tool_calls": ["rag_search"],
                        "citations": [{"doc_name": "陕西规则.pdf", "excerpt": "..."}],
                        "confidence": 0.85,
                        "trace_id": "trace_abc123"
                    }
                }
            }
        }
    }
)
def chat(
    req: AgentRequest,
    db: Session = Depends(get_db),
    dialog_manager = Depends(get_dialog_manager),
) -> AgentResponse:
    """
    Process query through PowerPolicyAgent.
    
    - Performs coreference resolution using conversation history
    - Routes query to appropriate tools based on intent
    - Records conversation turn for future context
    """
    trace_id = f"trace_{uuid.uuid4().hex[:12]}"
    set_trace_id(trace_id)
    set_session_id(req.session_id)
    
    resolved_query = dialog_manager.resolve_query(req.query, req.session_id)
    
    history = dialog_manager.get_history(req.session_id)
    
    try:
        agent = agent_singleton.get_agent(db)
        
        enhanced_req = AgentRequest(
            query=resolved_query,
            session_id=req.session_id,
            province_codes=req.province_codes,
            history=history,
            context=req.context,
        )
        
        response = agent.chat(enhanced_req)
        response.trace_id = trace_id
        
        dialog_manager.append_turn(
            session_id=req.session_id,
            query=req.query,
            reply=response.answer,
            intent=response.intent,
            province_code=response.detected_provinces,
        )
        
        return response
        
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except RuntimeError as exc:
        if "not preloaded" in str(exc):
            raise HTTPException(
                status_code=503,
                detail="Agent not ready. Please try again later."
            ) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/benchmark",
    response_model=BenchmarkResponse,
    summary="Get Benchmark Questions",
    description="Get random benchmark questions for testing.",
)
def get_benchmark_questions(
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResponse:
    questions = benchmark_service.get_random_questions(count=5)
    return BenchmarkResponse(
        questions=[
            {
                "question_id": q.get("question_id"),
                "question": q.get("question", ""),
                "category": q.get("category"),
            }
            for q in questions
        ]
    )


@router.post(
    "/title",
    response_model=TitleResponse,
    summary="Generate Session Title",
    description="Generate a title for a conversation session based on its history.",
)
def generate_title(
    req: TitleRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
    title_generator: TitleGenerator = Depends(get_title_generator),
) -> TitleResponse:
    existing_title = conversation_service.get_title(req.session_id)
    if existing_title:
        return TitleResponse(
            session_id=req.session_id,
            title=existing_title,
            generated=False,
        )
    
    turn_count = conversation_service.get_turn_count(req.session_id)
    if turn_count < 1:
        return TitleResponse(
            session_id=req.session_id,
            title="新对话",
            generated=False,
        )
    
    history = conversation_service.get_history(req.session_id)
    title = title_generator.generate(history)
    
    conversation_service.update_title(req.session_id, title)
    
    return TitleResponse(
        session_id=req.session_id,
        title=title,
        generated=True,
    )