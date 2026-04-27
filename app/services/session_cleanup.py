from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class SessionCleanupTask:
    def __init__(self, interval_minutes: int = 30, ttl_minutes: int = 120):
        self.interval_minutes = interval_minutes
        self.ttl_minutes = ttl_minutes
        self._running = False
        self._thread = None
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Session cleanup task started (interval={self.interval_minutes}min, ttl={self.ttl_minutes}min)")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _run_loop(self):
        while self._running:
            try:
                self._cleanup()
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
            
            time.sleep(self.interval_minutes * 60)
    
    def _cleanup(self):
        from app.db.repositories.conversation_repo import ConversationRepository
        from app.db.repositories.conversation_turn_repo import ConversationTurnRepository
        from app.db.repositories.trace_repo import TraceRepository
        from app.db.session import SessionLocal

        with SessionLocal() as db:
            state_repo = ConversationRepository(db)
            state_repo.clear_expired(self.ttl_minutes)

            turn_repo = ConversationTurnRepository(db)
            turn_repo.clear_expired(self.ttl_minutes)

            trace_repo = TraceRepository(db)
            trace_count = trace_repo.clear_expired(self.ttl_minutes)

            db.commit()
            logger.info(f"Expired sessions and turns cleared (ttl={self.ttl_minutes}min, traces={trace_count})")


cleanup_task = SessionCleanupTask()