"""腾讯云通用印刷体OCR请求与结果解析。"""
import base64
import hashlib
import hmac
import json
import logging
import time
from urllib.parse import urlparse

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)
MAX_BASE64_IMAGE_BYTES = 10 * 1024 * 1024


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hmac_sha256(key: str | bytes, value: str) -> bytes:
    if isinstance(key, str):
        key = key.encode("utf-8")
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()


def _authorization(payload: str, host: str, action: str, timestamp: int) -> str:
    algorithm = "TC3-HMAC-SHA256"
    service = "ocr"
    date = time.strftime("%Y-%m-%d", time.gmtime(timestamp))
    canonical_headers = (
        "content-type:application/json; charset=utf-8\n"
        f"host:{host}\n"
        f"x-tc-action:{action.lower()}\n"
    )
    signed_headers = "content-type;host;x-tc-action"
    canonical_request = "\n".join((
        "POST", "/", "", canonical_headers, signed_headers, _sha256(payload),
    ))
    credential_scope = f"{date}/{service}/tc3_request"
    string_to_sign = "\n".join((
        algorithm, str(timestamp), credential_scope, _sha256(canonical_request),
    ))
    secret_date = _hmac_sha256(f"TC3{settings.TENCENT_OCR_SECRET_KEY}", date)
    secret_service = _hmac_sha256(secret_date, service)
    secret_signing = _hmac_sha256(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    return (
        f"{algorithm} Credential={settings.TENCENT_OCR_SECRET_ID}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )


def recognize_image(image: bytes) -> str:
    """把单页PNG发送到腾讯OCR，返回按识别顺序拼接的文字。"""
    if not image:
        raise ValueError("OCR图片为空")
    if ((len(image) + 2) // 3) * 4 > MAX_BASE64_IMAGE_BYTES:
        raise ValueError("扫描页图片经Base64编码后超过腾讯OCR的10MB限制，请降低OCR_DPI后重试")
    if not settings.TENCENT_OCR_SECRET_ID or not settings.TENCENT_OCR_SECRET_KEY:
        raise ValueError("腾讯OCR未配置，请设置TENCENT_OCR_SECRET_ID和TENCENT_OCR_SECRET_KEY")

    endpoint = settings.TENCENT_OCR_ENDPOINT.strip()
    parsed = urlparse(endpoint)
    if (parsed.scheme != "https" or parsed.netloc != "ocr.tencentcloudapi.com"
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise ValueError("腾讯OCR地址必须是官方HTTPS接口地址")
    host = parsed.netloc
    encoded_image = base64.b64encode(image)
    payload = json.dumps(
        {"ImageBase64": encoded_image.decode("ascii")},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    timestamp = int(time.time())
    action = settings.TENCENT_OCR_ACTION
    headers = {
        "Authorization": _authorization(payload, host, action, timestamp),
        "Content-Type": "application/json; charset=utf-8",
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": settings.TENCENT_OCR_VERSION,
        "X-TC-Region": settings.TENCENT_OCR_REGION,
    }
    with httpx.Client(timeout=settings.TENCENT_OCR_TIMEOUT_SECONDS, verify=True) as client:
        response = client.post(endpoint, content=payload.encode("utf-8"), headers=headers)
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError("腾讯OCR返回内容不是有效JSON") from exc
    body = data.get("Response") if isinstance(data, dict) else None
    if not isinstance(body, dict):
        raise ValueError("腾讯OCR返回格式异常")
    error = body.get("Error")
    if isinstance(error, dict):
        code = str(error.get("Code") or "UNKNOWN_ERROR")
        logger.warning("腾讯OCR请求失败，错误代码：%s", code)
        raise ValueError(f"腾讯OCR调用失败（{code}），请检查密钥、区域和服务配额")

    detections = body.get("TextDetections", [])
    if not isinstance(detections, list):
        raise ValueError("腾讯OCR返回的文字列表格式异常")
    lines = [
        str(item.get("DetectedText") or "").strip()
        for item in detections
        if isinstance(item, dict) and str(item.get("DetectedText") or "").strip()
    ]
    logger.info("腾讯OCR识别出%d行文字", len(lines))
    return "\n".join(lines)
