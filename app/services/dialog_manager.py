from __future__ import annotations

from typing import Callable, List

from sqlalchemy.orm import Session

from app.services.conversation_service import ConversationService
from app.services.coreference_resolver import CoreferenceResolver


class DialogManager:
    """
    Unified dialog manager for multi-channel conversation handling.
    
    Provides standardized session creation, history retrieval, coreference resolution,
    and turn recording. Reuses existing ConversationService and CoreferenceResolver.
    """
    
    def __init__(
        self,
        session_factory: Callable[[], Session],
        enable_coreference: bool = True,
    ):
        self.conversation_service = ConversationService(session_factory)
        self.coreference_resolver = CoreferenceResolver(enabled=enable_coreference)
    
    def create_session(self, user_id: str, channel: str) -> str:
        """
        Create session ID in format: {channel}:{user_id}
        
        Args:
            user_id: User identifier (e.g., open_id, user_id)
            channel: Channel name (e.g., "feishu", "wechat", "api")
            
        Returns:
            Session ID string
        """
        return f"{channel}:{user_id}"
    
    def get_history(self, session_id: str) -> List[str]:
        """
        Get conversation history for a session.
        
        Returns format: ["Q: xxx", "A: xxx", ...]
        Automatically compresses to summary + recent 4 turns when needed.
        """
        return self.conversation_service.get_history(session_id)
    
    def resolve_query(self, query: str, session_id: str) -> str:
        """
        Resolve coreferences in query using conversation history.
        
        Replaces references like "那个政策", "它" with specific entities
        extracted from history.
        """
        history = self.get_history(session_id)
        return self.coreference_resolver.resolve(query, history)
    
    def append_turn(
        self,
        session_id: str,
        query: str,
        reply: str,
        intent: str = None,
        province_code: str = None,
        latency_ms: int = None,
    ) -> None:
        """
        Record a conversation turn.
        
        Args:
            session_id: Session identifier
            query: User query
            reply: Bot reply
            intent: Detected intent
            province_code: Province code if detected
            latency_ms: Response latency in milliseconds
        """
        self.conversation_service.append_turn(
            session_id=session_id,
            user_query=query,
            bot_reply=reply,
            intent=intent,
            province_code=province_code,
            latency_ms=latency_ms,
        )
    
    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self.conversation_service.clear_history(session_id)