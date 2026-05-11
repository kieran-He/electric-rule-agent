from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Callable

import lark_oapi as lark
from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody, ReplyMessageRequest, ReplyMessageRequestBody
from sqlalchemy.orm import Session

from app.agent.agent_singleton import agent_singleton
from app.core.logging_context import set_trace_id, set_session_id
from app.db.models.processed_message import ProcessedMessage
from app.schemas.agent import AgentRequest, AgentResponse
from app.services.conversation_service import ConversationService
from app.services.coreference_resolver import CoreferenceResolver
from app.services.trace_service import TraceService
from app.utils.markdown_to_feishu import MarkdownToFeishuConverter

logger = logging.getLogger(__name__)


class FeishuAgentService:
    """
    Feishu bot service using Agent architecture.
    
    Message flow: Feishu message -> Agent -> Tools -> Response
    """
    
    def __init__(
        self,
        settings,
        session_factory: Callable[[], Session],
        client: lark.Client,
    ):
        self.settings = settings
        self.session_factory = session_factory
        self.client = client
        self.conversation_service = ConversationService(session_factory)
        self.coreference_resolver = CoreferenceResolver(
            enabled=settings.coreference_resolution_enabled
        )
        self._processed_ttl = 86400
        self._md_converter = MarkdownToFeishuConverter()
    
    def _get_session_id(self, open_id: str) -> str:
        return f"feishu:{open_id}"
    
    def _get_event_id(self, data: lark.im.v1.P2ImMessageReceiveV1) -> str:
        if data.header and data.header.event_id:
            return data.header.event_id
        return data.uuid or ""
    
    def _check_and_mark(self, event_id: str) -> bool:
        if not event_id:
            return False
        with self.session_factory() as session:
            try:
                existing = session.query(ProcessedMessage).filter(
                    ProcessedMessage.event_id == event_id
                ).first()
                if existing:
                    return True
                
                cutoff = datetime.utcnow() - timedelta(seconds=self._processed_ttl)
                session.query(ProcessedMessage).filter(
                    ProcessedMessage.created_at < cutoff
                ).delete()
                
                session.add(ProcessedMessage(event_id=event_id))
                session.commit()
                return False
            except Exception:
                session.rollback()
                raise
    
    def _extract_message_text(self, message: lark.im.v1.Message) -> str:
        content = message.content
        if not content:
            return ""
        try:
            content_dict = json.loads(content)
            text = content_dict.get("text", "")
        except json.JSONDecodeError:
            text = content
        text = self._remove_mention(text)
        return text.strip()
    
    def _remove_mention(self, text: str) -> str:
        text = re.sub(r'<at user_id="[^"]*"[^>]*>@[^<]+</at>', '', text)
        text = re.sub(r'@[\w\u4e00-\u9fff]+', '', text)
        return text.strip()
    
    def _format_reply(self, response: AgentResponse) -> str:
        return response.answer
    
    def handle_message(self, data: lark.im.v1.P2ImMessageReceiveV1) -> None:
        if not data.event or not data.event.message:
            return
        
        message = data.event.message
        message_id = message.message_id
        
        open_id = ""
        if data.event.sender and data.event.sender.sender_id:
            open_id = data.event.sender.sender_id.open_id or ""
        
        session_id = self._get_session_id(open_id)
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        
        set_trace_id(trace_id)
        set_session_id(session_id)
        
        event_id = self._get_event_id(data)
        if not event_id:
            event_id = message_id
            logger.warning(f"No event_id found, using message_id as fallback: {message_id}")
        
        if self._check_and_mark(event_id):
            logger.debug(f"Duplicate event {event_id}, skipping")
            return
        
        chat_type = message.chat_type
        text = self._extract_message_text(message)
        
        if not text:
            logger.debug("Empty message text, skipping")
            return
        
        logger.info(f"Received message from {open_id} in {chat_type}: {text[:100]}")
        
        reply_msg_id = self._reply_message(message_id, "正在思考中，请稍候...")
        
        try:
            response = self._process_with_agent(text, session_id, trace_id)
            reply_text = self._format_reply(response)
            
            self.conversation_service.append_turn(
                session_id=session_id,
                user_query=text,
                bot_reply=response.answer,
                intent=response.intent,
            )
        except Exception as e:
            logger.exception(f"Error processing query: {e}")
            reply_text = "抱歉，处理您的请求时出现错误，请稍后重试。"
        
        if reply_msg_id:
            self._update_message(reply_msg_id, reply_text)
        else:
            self._reply_message(message_id, reply_text)
    
    def _process_with_agent(self, text: str, session_id: str, trace_id: str) -> AgentResponse:
        history = self.conversation_service.get_history(session_id)
        
        resolved_query = self.coreference_resolver.resolve(text, history)
        if resolved_query != text:
            logger.info(f"Coreference resolved: '{text}' -> '{resolved_query}'")
        
        with self.session_factory() as db:
            trace_service = TraceService(self.session_factory)
            agent = agent_singleton.get_agent(db)
            
            request = AgentRequest(
                query=resolved_query,
                session_id=session_id,
                province_codes=self.settings.province_defaults,
                history=history,
            )
            
            response = agent.chat(request, db=db, trace_service=trace_service)
            
            db.commit()
            return response
    
    def _reply_message(self, message_id: str, text: str) -> str | None:
        card_content = self._md_converter.convert_to_interactive(text)
        
        request = ReplyMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(ReplyMessageRequestBody.builder()
                          .msg_type("interactive")
                          .content(json.dumps(card_content))
                          .build()) \
            .build()
        
        try:
            response = self.client.im.v1.message.reply(request)
            if response.success() and response.data:
                returned_msg_id = response.data.message_id
                parent_id = getattr(response.data, 'parent_id', None)
                root_id = getattr(response.data, 'root_id', None)
                create_time = getattr(response.data, 'create_time', None)
                logger.info(f"Reply API response: new_msg_id={returned_msg_id}, parent_id={parent_id}, root_id={root_id}, create_time={create_time}, input_msg_id={message_id}")
                if response.raw and response.raw.content:
                    logger.debug(f"Reply API raw response: {response.raw.content[:500]}")
                if returned_msg_id == message_id:
                    logger.warning("Reply API returned same message_id as input - unexpected behavior")
                return returned_msg_id
            else:
                logger.error(f"Reply failed: {response.code} - {response.msg}")
                return None
        except Exception as e:
            logger.exception(f"Error sending reply: {e}")
            return None
    
    def _update_message(self, message_id: str, text: str) -> bool:
        card_content = self._md_converter.convert_to_interactive(text)
        
        request = PatchMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(PatchMessageRequestBody.builder()
                          .content(json.dumps(card_content))
                          .build()) \
            .build()
        
        try:
            response = self.client.im.v1.message.patch(request)
            if response.success():
                logger.info(f"Message {message_id} updated successfully")
                return True
            else:
                logger.error(f"Update failed: {response.code} - {response.msg}")
                return False
        except Exception as e:
            logger.exception(f"Error updating message: {e}")
            return False