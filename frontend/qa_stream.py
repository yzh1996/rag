"""解析问答 SSE 消息，保持中文、换行以及结束状态的完整性。"""
import json


def read_answer_events(response):
    """按 SSE 空行边界解析事件；HTTP 分块边界可能落在任意字符或消息内部。"""
    response.encoding = "utf-8"
    event, data = "message", []
    # 小块读取避免 requests 默认缓冲把前几个答案片段攒到一起才交给页面。
    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line:
            if data:
                payload = json.loads("\n".join(data))
                if not isinstance(payload, dict):
                    raise ValueError("流式响应格式异常")
                yield {"event": event, "data": payload}
            event, data = "message", []
            continue
        if line.startswith(":"):
            continue  # 心跳不进入回答气泡，也不影响首字耗时。
        field, separator, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if separator and field == "event":
            event = value
        elif separator and field == "data":
            data.append(value)
    if data:
        raise ValueError("连接中断，流式消息未接收完整")
