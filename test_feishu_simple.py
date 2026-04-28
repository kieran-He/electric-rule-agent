import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

import lark_oapi as lark
from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody

APP_ID = "cli_a964764e7efadbdd"
APP_SECRET = "mYKAuluFp9lKCM6PLH3HGb8Auom0F4wv"
LOG_LEVEL = lark.LogLevel.DEBUG

client = lark.Client.builder() \
    .app_id(APP_ID) \
    .app_secret(APP_SECRET) \
    .log_level(LOG_LEVEL) \
    .build()


def do_p2_im_message_receive_v1(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
    print(f"[do_p2_im_message_receive_v1] data: {lark.JSON.marshal(data, indent=4)}")

    if data.event and data.event.message:
        message = data.event.message
        message_id = message.message_id
        reply_text = "收到！"

        request = ReplyMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(ReplyMessageRequestBody.builder()
                          .msg_type("text")
                          .content('{"text":"' + reply_text + '"}')
                          .build()) \
            .build()

        try:
            response = client.im.v1.message.reply(request)
            print(f"回复成功: {lark.JSON.marshal(response)}")
        except Exception as e:
            print(f"回复失败: {e}")


event_handler = lark.EventDispatcherHandler.builder("", "") \
    .register_p2_im_message_receive_v1(do_p2_im_message_receive_v1) \
    .build()


def main():
    cli = lark.ws.Client(APP_ID, APP_SECRET,
                         event_handler=event_handler,
                         log_level=LOG_LEVEL)
    print("Starting Feishu WebSocket client...")
    cli.start()


if __name__ == "__main__":
    main()