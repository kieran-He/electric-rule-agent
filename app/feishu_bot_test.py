import lark_oapi as lark
from lark_oapi.api.im.v1 import *

# 配置常量
# 应用ID
APP_ID = "cli_a964764e7efadbdd"
# 应用密钥
APP_SECRET = "mYKAuluFp9lKCM6PLH3HGb8Auom0F4wv"
# 日志级别
LOG_LEVEL = lark.LogLevel.DEBUG

# 全局客户端
client = lark.Client.builder() \
    .app_id(APP_ID) \
    .app_secret(APP_SECRET) \
    .log_level(LOG_LEVEL) \
    .build()


## P2ImMessageReceiveV1 为接收消息 v2.0；CustomizedEvent 内的 message 为接收消息 v1.0。
def do_p2_im_message_receive_v1(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
    print(f'[ do_p2_im_message_receive_v1 access ], data: {lark.JSON.marshal(data, indent=4)}')

    # 回复消息
    if data.event and data.event.message:
        message = data.event.message
        message_id = message.message_id

        # 构建回复内容
        reply_text = "收到！"

        # 构建回复消息请求
        request = ReplyMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(ReplyMessageRequestBody.builder()
                          .msg_type("text")
                          .content('{"text":"' + reply_text + '"}')
                          .build()) \
            .build()

        # 发送请求
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
    cli.start()


if __name__ == "__main__":
    main()
