import logging
import signal
import sys
from concurrent.futures import ThreadPoolExecutor

import lark_oapi as lark

from app.config import settings
from app.db.session import SessionLocal, init_db
from app.services.feishu_bot_service import FeishuBotService

logger = logging.getLogger(__name__)


class FeishuBot:
    def __init__(self):
        self.app_id = settings.feishu_app_id
        self.app_secret = settings.feishu_app_secret
        self.client: lark.Client | None = None
        self.service: FeishuBotService | None = None
        self.ws_client: lark.ws.Client | None = None
        self.executor = ThreadPoolExecutor(
            max_workers=settings.feishu_max_workers,
            thread_name_prefix="feishu-handler",
        )

    def _validate_config(self) -> bool:
        if not self.app_id or not self.app_secret:
            logger.error("FEISHU_APP_ID and FEISHU_APP_SECRET must be set")
            return False
        return True

    def _init_db(self) -> None:
        init_db()
        logger.info("Database initialized")

    def _init_client(self) -> None:
        self.client = lark.Client.builder() \
            .app_id(self.app_id) \
            .app_secret(self.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        self.service = FeishuBotService(
            settings=settings,
            session_factory=SessionLocal,
            client=self.client,
        )
        logger.info("Feishu client initialized")

    def _handle_message(self, data: lark.im.v1.P2ImMessageReceiveV1) -> None:
        if self.service:
            self.executor.submit(self.service.handle_message, data)

    def _create_event_handler(self) -> lark.EventDispatcherHandler:
        return lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(self._handle_message) \
            .build()

    def start(self) -> None:
        if not self._validate_config():
            sys.exit(1)
        
        self._init_db()
        self._init_client()
        
        event_handler = self._create_event_handler()
        
        self.ws_client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )
        
        logger.info("Starting Feishu bot WebSocket client...")
        try:
            self.ws_client.start()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            self.stop()
        except Exception as e:
            logger.exception(f"WebSocket client error: {e}")
            self.stop()
            raise

    def stop(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Feishu bot stopped")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    bot = FeishuBot()
    
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        bot.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    bot.start()


if __name__ == "__main__":
    main()