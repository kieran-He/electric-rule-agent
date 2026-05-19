from __future__ import annotations

import json
import logging
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.db.session import SessionLocal
import lark_oapi as lark

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/feishu",
    tags=["feishu"],
)

# 线程池处理异步任务
executor = ThreadPoolExecutor(
    max_workers=getattr(settings, 'feishu_max_workers', 10),
    thread_name_prefix="feishu-webhook-handler",
)

# 全局实例（懒加载）
_client: lark.Client | None = None
_feishu_service = None


def get_client() -> lark.Client:
    """获取 lark.Client 实例"""
    global _client
    if _client is None:
        _client = lark.Client.builder() \
            .app_id(settings.feishu_app_id) \
            .app_secret(settings.feishu_app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
    return _client


def get_feishu_service():
    """获取 FeishuAgentService 实例"""
    global _feishu_service
    if _feishu_service is None:
        from app.services.feishu_agent_service import FeishuAgentService
        _feishu_service = FeishuAgentService(
            settings=settings,
            session_factory=SessionLocal,
            client=get_client(),
        )
        logger.info("FeishuAgentService initialized for webhook mode")
    return _feishu_service


def verify_signature(timestamp: str, nonce: str, body: bytes, signature: str) -> bool:
    """验证飞书请求签名"""
    if not settings.feishu_app_secret:
        logger.warning("FEISHU_APP_SECRET not configured, skipping signature verification")
        return True
    
    # 验证时间戳（防止重放攻击）
    try:
        ts = int(timestamp)
        current_ts = int(time.time())
        if abs(current_ts - ts) > 300:  # 5分钟有效期
            logger.warning(f"Timestamp expired: {timestamp}, current: {current_ts}")
            return False
    except ValueError:
        logger.warning(f"Invalid timestamp: {timestamp}")
        return False
    
    # 计算签名
    token = settings.feishu_app_secret
    sign_str = f"{timestamp}{nonce}{token}"
    expected_signature = hashlib.sha256(sign_str.encode()).hexdigest()
    
    if signature != expected_signature:
        logger.warning(f"Signature mismatch: expected {expected_signature}, got {signature}")
        return False
    
    return True


def create_message_event(event_data: dict) -> lark.im.v1.P2ImMessageReceiveV1:
    """从JSON数据构造 P2ImMessageReceiveV1 对象"""
    header = event_data.get("header", {})
    event = event_data.get("event", {})
    message = event.get("message", {})
    sender = event.get("sender", {})
    sender_id = sender.get("sender_id", {})
    
    # 创建消息对象
    msg = lark.im.v1.Message()
    msg.message_id = message.get("message_id", "")
    msg.chat_id = message.get("chat_id", "")
    msg.chat_type = message.get("chat_type", "p2p")
    msg.content = message.get("content", "")
    msg.create_time = message.get("create_time", 0)
    
    # 创建发送者对象 - 使用 Sender 类
    s = lark.im.v1.Sender()
    s.open_id = sender_id.get("open_id", "")
    s.user_id = sender_id.get("user_id", "")
    s.union_id = sender_id.get("union_id", "")
    
    # 创建完整事件对象并直接设置属性
    data = lark.im.v1.P2ImMessageReceiveV1()
    
    # 设置 header 属性
    data.header = type('Header', (), {
        'event_id': header.get("event_id", ""),
        'event_type': header.get("event_type", ""),
        'create_time': header.get("create_time", 0),
        'token': header.get("token", ""),
        'app_id': header.get("app_id", ""),
    })()
    
    # 设置 event 属性
    data.event = type('Event', (), {
        'message': msg,
        'sender': type('EventSender', (), {'sender_id': s})(),
    })()
    
    return data


def create_card_event(event_data: dict) -> lark.Card:
    """从JSON数据构造 Card 对象"""
    event = event_data.get("event", {})
    action = event.get("action", {})
    
    # 创建 Card 对象
    card = lark.Card()
    card.open_message_id = event.get("open_message_id", "")
    card.open_chat_id = event.get("open_chat_id", "")
    card.open_id = event.get("open_id", "")
    
    # 创建 Action 对象
    act = lark.card.model.Action()
    act.value = action.get("value", {})
    act.tag = action.get("tag", "")
    
    card.action = act
    
    return card


def handle_message_task(event_data: dict):
    """异步处理消息事件"""
    try:
        service = get_feishu_service()
        data = create_message_event(event_data)
        service.handle_message(data)
    except Exception as e:
        logger.exception(f"Error handling message event: {e}")


def handle_card_task(event_data: dict):
    """异步处理卡片事件"""
    try:
        service = get_feishu_service()
        card = create_card_event(event_data)
        service.handle_card_action(card)
    except Exception as e:
        logger.exception(f"Error handling card action: {e}")


@router.post(
    "/webhook",
    summary="Feishu Webhook Endpoint",
    description="接收飞书推送的事件（消息、卡片回调等）",
)
async def feishu_webhook(request: Request):
    """
    飞书 Webhook 端点
    
    接收飞书推送的事件并异步处理：
    - 消息接收事件 (im.message.receive_v1)
    - 卡片回调事件 (card.action.trigger)
    """
    # 获取请求头
    timestamp = request.headers.get("X-Lark-Request-Timestamp", "")
    nonce = request.headers.get("X-Lark-Request-Nonce", "")
    signature = request.headers.get("X-Lark-Signature", "")
    
    # 获取请求体
    body = await request.body()
    
    # 解析事件
    try:
        event_data = json.loads(body)
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON body: {body[:200]}")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    # 处理 URL 验证请求（飞书首次配置时会发送，跳过签名验证）
    if event_data.get("type") == "url_verification":
        challenge = event_data.get("challenge", "")
        logger.info(f"URL verification request, challenge: {challenge}")
        return JSONResponse(content={"challenge": challenge})
    
    # 验证签名（对非 URL 验证请求）
    # 如果请求头无签名信息，跳过验证（允许测试）
    if signature and timestamp and nonce:
        if getattr(settings, 'feishu_verify_signature', True) and not verify_signature(timestamp, nonce, body, signature):
            raise HTTPException(status_code=403, detail="Signature verification failed")
    else:
        logger.debug("Skipping signature verification (no signature headers)")
    
    # 解析事件类型
    header = event_data.get("header", {})
    event_type = header.get("event_type", "")
    event_id = header.get("event_id", "")
    
    logger.info(f"Received feishu event: type={event_type}, event_id={event_id}")
    
    # 根据事件类型处理
    if event_type == "im.message.receive_v1":
        executor.submit(handle_message_task, event_data)
    
    elif event_type == "card.action.trigger":
        executor.submit(handle_card_task, event_data)
    
    else:
        logger.warning(f"Unknown event type: {event_type}")
    
    # 返回成功响应（飞书要求在3秒内返回）
    return JSONResponse(content={"code": 0, "msg": "success"})


@router.get(
    "/health",
    summary="Feishu Webhook Health Check",
)
async def feishu_health():
    """检查飞书 Webhook 服务状态"""
    return {
        "status": "ok",
        "service_loaded": _feishu_service is not None,
        "client_loaded": _client is not None,
        "executor_workers": getattr(settings, 'feishu_max_workers', 10),
    }