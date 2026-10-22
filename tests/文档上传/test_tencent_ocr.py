import base64
import json

import pytest

from config.settings import settings
from core import content_processing, tencent_ocr


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


@pytest.fixture
def tencent_credentials(monkeypatch):
    monkeypatch.setattr(settings, "TENCENT_OCR_SECRET_ID", "test-secret-id")
    monkeypatch.setattr(settings, "TENCENT_OCR_SECRET_KEY", "test-secret-key")
    monkeypatch.setattr(settings, "TENCENT_OCR_ENDPOINT", "https://ocr.tencentcloudapi.com")
    monkeypatch.setattr(settings, "TENCENT_OCR_REGION", "ap-guangzhou")
    monkeypatch.setattr(settings, "TENCENT_OCR_ACTION", "GeneralBasicOCR")
    monkeypatch.setattr(settings, "TENCENT_OCR_VERSION", "2018-11-19")
    monkeypatch.setattr(settings, "TENCENT_OCR_TIMEOUT_SECONDS", 15)


def test_tencent_request_is_signed_and_text_lines_are_returned(monkeypatch, tencent_credentials):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, *, content, headers):
            captured.update(url=url, content=content, headers=headers)
            return FakeResponse({"Response": {"TextDetections": [
                {"DetectedText": "\u5458\u5de5\u5165\u804c"}, {"DetectedText": "\u6ee1\u4e00\u5e74"}, {"DetectedText": " "},
            ]}})

    monkeypatch.setattr(tencent_ocr.httpx, "Client", FakeClient)
    result = tencent_ocr.recognize_image(b"png-image")

    assert result == "\u5458\u5de5\u5165\u804c\n\u6ee1\u4e00\u5e74"
    assert captured["url"] == "https://ocr.tencentcloudapi.com"
    assert json.loads(captured["content"])["ImageBase64"] == base64.b64encode(b"png-image").decode("ascii")
    assert captured["headers"]["X-TC-Action"] == "GeneralBasicOCR"
    assert captured["headers"]["X-TC-Region"] == "ap-guangzhou"
    assert captured["headers"]["Authorization"].startswith("TC3-HMAC-SHA256 Credential=test-secret-id/")
    assert captured["client"] == {"timeout": 15, "verify": True}


def test_tencent_request_requires_credentials_before_network(monkeypatch, tencent_credentials):
    monkeypatch.setattr(settings, "TENCENT_OCR_SECRET_ID", "")
    monkeypatch.setattr(tencent_ocr.httpx, "Client", lambda **kwargs: pytest.fail("network client must not be created"))

    with pytest.raises(ValueError, match="TENCENT_OCR_SECRET_ID"):
        tencent_ocr.recognize_image(b"png-image")


def test_tencent_request_checks_base64_encoded_size(monkeypatch, tencent_credentials):
    monkeypatch.setattr(tencent_ocr.httpx, "Client", lambda **kwargs: pytest.fail("network client must not be created"))

    raw_size = (tencent_ocr.MAX_BASE64_IMAGE_BYTES * 3 // 4) + 1
    with pytest.raises(ValueError, match="10MB"):
        tencent_ocr.recognize_image(b"x" * raw_size)


def test_tencent_request_rejects_untrusted_endpoint(monkeypatch, tencent_credentials):
    monkeypatch.setattr(settings, "TENCENT_OCR_ENDPOINT", "https://attacker.example.com")
    monkeypatch.setattr(tencent_ocr.httpx, "Client", lambda **kwargs: pytest.fail("network client must not be created"))

    with pytest.raises(ValueError):
        tencent_ocr.recognize_image(b"png-image")


def test_tencent_api_error_does_not_expose_credentials(monkeypatch, tencent_credentials):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse({"Response": {"Error": {
                "Code": "AuthFailure.SecretIdNotFound", "Message": "sensitive service detail",
            }}})

    monkeypatch.setattr(tencent_ocr.httpx, "Client", FakeClient)

    with pytest.raises(ValueError, match="AuthFailure.SecretIdNotFound") as error:
        tencent_ocr.recognize_image(b"png-image")
    assert "test-secret-key" not in str(error.value)
    assert "sensitive service detail" not in str(error.value)


def test_ocr_provider_selects_tencent_or_local_tesseract(monkeypatch):
    monkeypatch.setattr(settings, "OCR_ENABLED", True)
    monkeypatch.setattr(settings, "OCR_PROVIDER", "tencent")
    monkeypatch.setattr(content_processing, "_ocr_page_tencent", lambda page: "cloud text")
    monkeypatch.setattr(content_processing, "_ocr_page_tesseract", lambda page: "local text")
    assert content_processing.ocr_page(object()) == "cloud text"

    monkeypatch.setattr(settings, "OCR_PROVIDER", "tesseract")
    assert content_processing.ocr_page(object()) == "local text"


def test_tencent_failure_does_not_fall_back_to_tesseract(monkeypatch):
    monkeypatch.setattr(settings, "OCR_ENABLED", True)
    monkeypatch.setattr(settings, "OCR_PROVIDER", "tencent")
    monkeypatch.setattr(content_processing, "_ocr_page_tencent", lambda page: (_ for _ in ()).throw(ValueError("cloud failed")))
    monkeypatch.setattr(content_processing, "_ocr_page_tesseract", lambda page: pytest.fail("automatic local fallback is disabled"))

    with pytest.raises(ValueError, match="cloud failed"):
        content_processing.ocr_page(object())

def test_ocr_page_renders_pdf_page_before_tencent_call(monkeypatch):
    from core import tencent_ocr

    monkeypatch.setattr(settings, "OCR_ENABLED", True)
    monkeypatch.setattr(settings, "OCR_PROVIDER", "tencent")
    captured = {}

    class Pixmap:
        def tobytes(self, image_format):
            captured["format"] = image_format
            return b"rendered-png"

    class Page:
        def get_pixmap(self, *, dpi, alpha):
            captured["render"] = {"dpi": dpi, "alpha": alpha}
            return Pixmap()

    def recognize(image):
        captured["image"] = image
        return "recognized page"

    monkeypatch.setattr(tencent_ocr, "recognize_image", recognize)
    assert content_processing.ocr_page(Page()) == "recognized page"
    assert captured == {
        "render": {"dpi": settings.OCR_DPI, "alpha": False},
        "format": "png",
        "image": b"rendered-png",
    }


def test_disabled_ocr_refuses_scanned_page(monkeypatch):
    monkeypatch.setattr(settings, "OCR_ENABLED", False)
    monkeypatch.setattr(content_processing, "_ocr_page_tencent", lambda page: pytest.fail("OCR must stay disabled"))

    with pytest.raises(ValueError, match="\u542f\u7528OCR"):
        content_processing.ocr_page(object())