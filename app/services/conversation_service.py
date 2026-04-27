from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

from sqlalchemy.orm import Session

from app.db.models.conversation_state import ConversationState
from app.db.models.conversation_turn import ConversationTurn
from app.db.repositories.conversation_repo import ConversationRepository
from app.db.repositories.conversation_turn_repo import ConversationTurnRepository
from app.services.history_summarizer import HistorySummarizer

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(
        self,
        session_factory: Callable[[], Session],
        max_history_turns: int = 4,
        enable_summary: bool = True,
    ):
        self.session_factory = session_factory
        self.max_history_turns = max_history_turns
        self.enable_summary = enable_summary
        self.summarizer = HistorySummarizer() if enable_summary else None

    def get_or_create(self, session_id: str) -> ConversationState:
        with self.session_factory() as db:
            repo = ConversationRepository(db)
            state = repo.get(session_id)
            if state is None:
                state = ConversationState(session_id=session_id)
                db.add(state)
                db.commit()
                db.refresh(state)
            return state

    def get_history(self, session_id: str) -> list[str]:
        """
        Get conversation history with compression.
        
        Returns:
            - 4 turns or less: full history ["Q: xxx", "A: xxx", ...]
            - More than 4 turns: summary + recent 4 turns
        """
        with self.session_factory() as db:
            turn_repo = ConversationTurnRepository(db)
            
            all_turns = turn_repo.get_turns(session_id, limit=100)
            all_turns = sorted(all_turns, key=lambda t: t.turn_index)
            
            if len(all_turns) <= self.max_history_turns:
                history = []
                for t in all_turns:
                    history.append(f"Q: {t.user_query}")
                    history.append(f"A: {t.bot_reply}")
                return history
            
            else:
                state_repo = ConversationRepository(db)
                state = state_repo.get(session_id)
                
                if state and state.history_summary:
                    summary = state.history_summary
                else:
                    old_turns = all_turns[:-self.max_history_turns]
                    summary = self._generate_summary(old_turns)
                    
                    if state:
                        state.history_summary = summary
                        state_repo.upsert(state)
                        db.commit()
                
                recent_turns = all_turns[-self.max_history_turns:]
                
                history = [f"【历史摘要】{summary}"]
                for t in recent_turns:
                    history.append(f"Q: {t.user_query}")
                    history.append(f"A: {t.bot_reply}")
                
                return history

    def _generate_summary(self, turns: list[ConversationTurn]) -> str:
        """Generate summary from old conversation turns."""
        if not turns:
            return ""
        
        if self.summarizer and self.summarizer.is_available():
            try:
                return self.summarizer.summarize(turns)
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")
        
        queries = [t.user_query[:30] for t in turns[:5]]
        return f"用户询问：{', '.join(queries)}"

    def append_turn(
        self,
        session_id: str,
        user_query: str,
        bot_reply: str,
        intent: str = None,
        province_code: str = None,
        latency_ms: int = None,
    ):
        """
        Append new turn and update history summary if needed.
        
        When conversation exceeds 4 turns, generate summary for old turns.
        """
        with self.session_factory() as db:
            turn_repo = ConversationTurnRepository(db)
            state_repo = ConversationRepository(db)

            turn_index = turn_repo.count_turns(session_id) + 1

            turn = ConversationTurn(
                session_id=session_id,
                turn_index=turn_index,
                user_query=user_query,
                bot_reply=bot_reply,
                intent=intent,
                province_code=province_code,
                latency_ms=latency_ms,
            )
            turn_repo.add_turn(turn)

            state = state_repo.get(session_id)
            if state is None:
                state = ConversationState(session_id=session_id)
            if province_code:
                state.province_code = province_code
            state.last_question = user_query
            state.last_intent = intent
            state.updated_at = datetime.utcnow()
            
            if turn_index > self.max_history_turns and self.enable_summary:
                all_turns = turn_repo.get_turns(session_id, limit=100)
                all_turns = sorted(all_turns, key=lambda t: t.turn_index)
                
                old_turns = all_turns[:-self.max_history_turns]
                
                if old_turns:
                    summary = self._generate_summary(old_turns)
                    state.history_summary = summary
            
            state_repo.upsert(state)

            db.commit()

    def update_context(self, session_id: str, province_code: str = None, market_type: str = None):
        with self.session_factory() as db:
            repo = ConversationRepository(db)
            state = repo.get(session_id)
            if state is None:
                state = ConversationState(session_id=session_id)
            if province_code:
                state.province_code = province_code
            if market_type:
                state.market_type = market_type
            repo.upsert(state)
            db.commit()

    def clear_history(self, session_id: str):
        """Clear conversation history (turns + summary)."""
        with self.session_factory() as db:
            turn_repo = ConversationTurnRepository(db)
            turn_repo.clear_turns(session_id)
            
            state_repo = ConversationRepository(db)
            state = state_repo.get(session_id)
            if state:
                state.history_summary = None
                state_repo.upsert(state)
            
            db.commit()