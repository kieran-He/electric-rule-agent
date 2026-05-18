from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable

from sqlalchemy.orm import Session

from app.db.models.langgraph_checkpoint import LangGraphCheckpoint

logger = logging.getLogger(__name__)


class CheckpointCleanupService:
    """
    Service for cleaning up old LangGraph checkpoints.
    
    Periodically removes checkpoints older than a specified number of days
    to prevent database bloat.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """
        Delete checkpoints older than N days.
        
        Args:
            days: Number of days to keep checkpoints (default: 7)
        
        Returns:
            Number of deleted checkpoints
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with self.session_factory() as db:
            deleted = db.query(LangGraphCheckpoint).filter(
                LangGraphCheckpoint.created_at < cutoff
            ).delete()
            db.commit()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} checkpoints older than {days} days")
            
            return deleted

    def cleanup_by_thread(self, thread_id: str) -> int:
        """
        Delete all checkpoints for a specific thread/session.
        
        Args:
            thread_id: The thread/session ID to clean up
        
        Returns:
            Number of deleted checkpoints
        """
        with self.session_factory() as db:
            deleted = db.query(LangGraphCheckpoint).filter(
                LangGraphCheckpoint.thread_id == thread_id
            ).delete()
            db.commit()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} checkpoints for thread {thread_id}")
            
            return deleted

    def get_checkpoint_count(self, thread_id: str | None = None) -> int:
        """
        Get the count of checkpoints.
        
        Args:
            thread_id: Optional thread ID to count checkpoints for
        
        Returns:
            Number of checkpoints
        """
        with self.session_factory() as db:
            query = db.query(LangGraphCheckpoint)
            
            if thread_id:
                query = query.filter(LangGraphCheckpoint.thread_id == thread_id)
            
            return query.count()