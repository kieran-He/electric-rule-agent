from __future__ import annotations

import base64
import json
import logging
from typing import Any, Callable, Iterator, TYPE_CHECKING

from sqlalchemy.orm import Session

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.constants import CONFIG_KEY_THREAD_ID

if TYPE_CHECKING:
    from langgraph.checkpoint.base import RunnableConfig, ChannelVersions

from app.db.models.langgraph_checkpoint import LangGraphCheckpoint

logger = logging.getLogger(__name__)


class DbCheckpointer(BaseCheckpointSaver):
    """
    Database-backed checkpointer for LangGraph.
    
    Stores checkpoints in PostgreSQL/MySQL for persistence across restarts.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory
        self.serde = JsonPlusSerializer()
    
    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> str:
        type_str, data_bytes = self.serde.dumps_typed(checkpoint)
        return json.dumps({
            "type": type_str,
            "data": base64.b64encode(data_bytes).decode("utf-8")
        })
    
    def _deserialize_checkpoint(self, data_str: str) -> Checkpoint:
        data = json.loads(data_str)
        type_str = data["type"]
        bytes_data = base64.b64decode(data["data"])
        return self.serde.loads_typed((type_str, bytes_data))

    def _get_thread_id(self, config: dict) -> str:
        configurable = config.get("configurable", {})
        thread_id = configurable.get(CONFIG_KEY_THREAD_ID, "default")
        return thread_id

    def _get_checkpoint_ns(self, config: dict) -> str:
        configurable = config.get("configurable", {})
        return configurable.get("checkpoint_ns", "")

    def _get_checkpoint_id(self, config: dict) -> str | None:
        configurable = config.get("configurable", {})
        return configurable.get("checkpoint_id")

    def get_tuple(self, config: dict) -> CheckpointTuple | None:
        """Get a checkpoint tuple from database."""
        thread_id = self._get_thread_id(config)
        checkpoint_ns = self._get_checkpoint_ns(config)
        checkpoint_id = self._get_checkpoint_id(config)

        with self.session_factory() as db:
            query = db.query(LangGraphCheckpoint).filter(
                LangGraphCheckpoint.thread_id == thread_id,
                LangGraphCheckpoint.checkpoint_ns == checkpoint_ns,
            )

            if checkpoint_id:
                query = query.filter(LangGraphCheckpoint.checkpoint_id == checkpoint_id)
            else:
                query = query.order_by(LangGraphCheckpoint.created_at.desc())

            checkpoint_record = query.first()

            if checkpoint_record:
                checkpoint_data = self._deserialize_checkpoint(checkpoint_record.checkpoint_data)
                metadata_data = json.loads(checkpoint_record.checkpoint_metadata or "{}")
                
                parent_config = None
                if checkpoint_record.parent_checkpoint_id:
                    parent_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_record.parent_checkpoint_id,
                        }
                    }

                return CheckpointTuple(
                    config=config,
                    checkpoint=checkpoint_data,
                    metadata=metadata_data,
                    parent_config=parent_config,
                )

        return None

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> dict:
        """Store a checkpoint in database."""
        thread_id = self._get_thread_id(config)
        checkpoint_ns = self._get_checkpoint_ns(config)
        checkpoint_id = checkpoint.get("id", "")

        checkpoint_data = self._serialize_checkpoint(checkpoint)
        metadata_data = json.dumps(metadata)

        with self.session_factory() as db:
            existing = db.query(LangGraphCheckpoint).filter(
                LangGraphCheckpoint.thread_id == thread_id,
                LangGraphCheckpoint.checkpoint_ns == checkpoint_ns,
                LangGraphCheckpoint.checkpoint_id == checkpoint_id,
            ).first()

            if existing:
                existing.checkpoint_data = checkpoint_data
                existing.checkpoint_metadata = metadata_data
            else:
                new_checkpoint = LangGraphCheckpoint(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    checkpoint_data=checkpoint_data,
                    checkpoint_metadata=metadata_data,
                )
                db.add(new_checkpoint)

            db.commit()

        return config

    def list(
        self,
        config: dict | None,
        *,
        filter: dict | None = None,
        before: dict | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from database."""
        if config is None:
            config = {}

        thread_id = self._get_thread_id(config)
        checkpoint_ns = self._get_checkpoint_ns(config)

        with self.session_factory() as db:
            query = db.query(LangGraphCheckpoint).filter(
                LangGraphCheckpoint.thread_id == thread_id,
                LangGraphCheckpoint.checkpoint_ns == checkpoint_ns,
            )

            if before:
                before_id = self._get_checkpoint_id(before)
                if before_id:
                    before_checkpoint = db.query(LangGraphCheckpoint).filter(
                        LangGraphCheckpoint.checkpoint_id == before_id
                    ).first()
                    if before_checkpoint:
                        query = query.filter(
                            LangGraphCheckpoint.created_at < before_checkpoint.created_at
                        )

            query = query.order_by(LangGraphCheckpoint.created_at.desc())

            if limit:
                query = query.limit(limit)

            for checkpoint_record in query:
                checkpoint_data = self._deserialize_checkpoint(checkpoint_record.checkpoint_data)
                metadata_data = json.loads(checkpoint_record.checkpoint_metadata or "{}")
                
                parent_config = None
                if checkpoint_record.parent_checkpoint_id:
                    parent_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_record.parent_checkpoint_id,
                        }
                    }

                yield CheckpointTuple(
                    config=config,
                    checkpoint=checkpoint_data,
                    metadata=metadata_data,
                    parent_config=parent_config,
                )

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread."""
        with self.session_factory() as db:
            db.query(LangGraphCheckpoint).filter(
                LangGraphCheckpoint.thread_id == thread_id
            ).delete()
            db.commit()

    def put_writes(
        self,
        config: dict,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store pending writes."""
        pass