"""基础模型请求计时的隔离验证，不访问真实模型服务。"""
import asyncio
import gzip
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI, DefaultHttpxClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from core import llm_http_timing as timing
from core.telemetry import span


@pytest.fixture
def recorded(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("http-timing-test")
    monkeypatch.setattr(trace, "get_tracer", lambda *args, **kwargs: tracer)
    yield exporter
    provider.shutdown()


class ScriptedStream(httpx.SyncByteStream):
    def __init__(self, chunks, clock=None, error=None):
        self.chunks = chunks
        self.clock = clock
        self.error = error
        self.closed = False

    def __iter__(self):
        for delay, chunk in self.chunks:
            if self.clock is not None:
                self.clock[0] += delay
            yield chunk
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True
        if self.clock is not None:
            self.clock[0] += 1


def by_name(recorded):
    return {item.name: item for item in recorded.get_finished_spans()}


def test_stream_times_headers_and_complete_data_event_without_read_ahead(monkeypatch, recorded):
    clock = [0.0]
    monkeypatch.setattr(timing, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    source = ScriptedStream([(3, b": heartbeat\n\n"), (4, b"da"),
        (1, 'ta: {"content":"正文"}\n'.encode()), (2, b"\n"), (3, b"data: [DONE]\n\n")], clock)
    def respond(request):
        clock[0] += 2
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=source)
    with timing.TimedHttpClient(transport=httpx.MockTransport(respond)) as client:
        with span("rag.generation") as parent, timing.http_timing_parent(parent):
            with span("rag.generation.wait_first_chunk"):
                response = client.send(client.build_request("POST", "https://provider.invalid/chat"), stream=True)
                assert clock[0] == 2 and not source.closed
                assert set(by_name(recorded)) == {"llm.http.wait_headers"}
                chunks = response.iter_bytes()
                assert next(chunks) == b": heartbeat\n\n"
                assert "llm.http.wait_first_event" not in by_name(recorded)
            assert trace.get_current_span() is parent
            content = b"".join(chunks)
            assert "正文" in content.decode() and source.closed
    spans = by_name(recorded)
    request = spans["llm.http.request"]
    assert request.parent.span_id == parent.get_span_context().span_id
    assert request.attributes["llm.http.headers_ms"] == 2000
    assert request.attributes["llm.http.first_event_ms"] == 12000
    assert request.attributes["llm.http.headers_to_first_event_ms"] == 10000
    assert request.attributes["llm.http.total_ms"] == 16000
    assert request.attributes["llm.http.response_complete"] is True
    assert request.attributes["llm.http.first_event_received"] is True
    assert [event.name for event in request.events] == ["request.started", "response.headers_received",
        "response.first_data_event_received", "request.finished"]
    assert spans["llm.http.wait_first_event"].parent.span_id == spans["llm.http.read_body"].context.span_id
    assert all(request.start_time <= item.start_time <= item.end_time <= request.end_time
        for name, item in spans.items() if name.startswith("llm.http."))
    assert "正文" not in json.dumps(dict(request.attributes), ensure_ascii=False)


@pytest.mark.parametrize("separator", [b"\n", b"\r\n", b"\r"])
@pytest.mark.parametrize("compressed", [False, True])
def test_split_compressed_sse_preserves_bytes_and_ignores_keepalive(recorded, separator, compressed):
    body = b": ping" + separator * 2 + b"event: message" + separator + 'data: 中文'.encode() + separator * 2
    wire = gzip.compress(body) if compressed else body
    source = ScriptedStream([(0, wire[index:index + 1]) for index in range(len(wire))])
    headers = {"content-type": "text/event-stream; charset=utf-8"}
    if compressed:
        headers["content-encoding"] = "gzip"
    with timing.TimedHttpClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, headers=headers, stream=source))) as client:
        with span("rag.generation"):
            response = client.send(client.build_request("POST", "https://provider.invalid/chat"), stream=True)
            assert response.read() == body
    request = by_name(recorded)["llm.http.request"]
    assert request.attributes["llm.http.first_event_received"] is True
    assert request.attributes["llm.http.response_complete"] is True and source.closed


def test_non_streaming_rewrite_measures_complete_body(recorded):
    source = ScriptedStream([(0, '{"answer":"完整问题"}'.encode())])
    with timing.TimedHttpClient(transport=httpx.MockTransport(lambda request: httpx.Response(200,
            headers={"content-type": "application/json"}, stream=source))) as client:
        with span("rag.rewrite") as parent:
            response = client.post("https://provider.invalid/chat", json={"question": "private question"})
            assert response.json() == {"answer": "完整问题"}
            assert source.closed
    spans = by_name(recorded)
    request = spans["llm.http.request"]
    assert request.parent.span_id == parent.get_span_context().span_id
    assert request.attributes["llm.http.response_complete"] is True
    assert "llm.http.wait_first_event" not in spans
    assert "llm.http.first_event_ms" not in request.attributes


def test_early_close_ends_all_spans_and_releases_response(recorded):
    source = ScriptedStream([(0, b": ping\n\n"), (0, b"data: private\n\n")])
    with timing.TimedHttpClient(transport=httpx.MockTransport(lambda request: httpx.Response(200,
            headers={"content-type": "text/event-stream"}, stream=source))) as client:
        with span("rag.generation"):
            response = client.send(client.build_request("POST", "https://provider.invalid/chat"), stream=True)
            assert next(response.iter_bytes()) == b": ping\n\n"
            response.close()
            response.close()
    spans = by_name(recorded)
    assert len(spans) == 5
    request = spans["llm.http.request"]
    assert request.attributes["llm.http.cancelled"] is True
    assert request.attributes["llm.http.first_event_received"] is False
    assert request.attributes["llm.http.response_complete"] is False and source.closed


@pytest.mark.parametrize("before_headers", [False, True])
def test_transport_errors_end_spans_without_exception_content(recorded, before_headers):
    source = ScriptedStream([], error=httpx.ReadError("private provider response"))
    def respond(request):
        if before_headers:
            raise httpx.ConnectError("private proxy credentials", request=request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=source)
    with timing.TimedHttpClient(transport=httpx.MockTransport(respond)) as client:
        with span("rag.generation"):
            with pytest.raises(httpx.TransportError):
                client.post("https://user:secret@provider.invalid/chat?api_key=secret")
    spans = by_name(recorded)
    request = spans["llm.http.request"]
    assert request.status.status_code == trace.StatusCode.ERROR
    assert request.attributes["error.type"] == ("ConnectError" if before_headers else "ReadError")
    assert all(item.end_time is not None for item in spans.values())
    exported = json.dumps([{"attributes": dict(item.attributes), "events": [dict(event.attributes) for event in item.events]}
        for item in spans.values()])
    assert "private" not in exported and "secret" not in exported
    if not before_headers:
        assert source.closed


def test_sdk_http_error_and_retry_behavior_are_preserved(recorded):
    attempts = []
    def respond(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(503, headers={"retry-after-ms": "1"}, json={"error": {"message": "private error"}})
        return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0, "model": "mock",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "改写结果"}, "finish_reason": "stop"}]})
    with timing.TimedHttpClient(transport=httpx.MockTransport(respond)) as client:
        model = OpenAI(api_key="test-key", http_client=client, max_retries=1)
        with span("rag.rewrite") as parent:
            answer = model.chat.completions.create(model="mock", messages=[{"role": "user", "content": "private question"}])
            assert answer.choices[0].message.content == "改写结果"
    requests = [item for item in recorded.get_finished_spans() if item.name == "llm.http.request"]
    assert len(attempts) == len(requests) == 2
    assert all(item.parent.span_id == parent.get_span_context().span_id for item in requests)
    assert requests[0].attributes["http.response.status_code"] == 503
    assert requests[0].status.status_code == trace.StatusCode.ERROR
    assert requests[1].attributes["http.response.status_code"] == 200


def test_observer_failure_does_not_break_answer(monkeypatch, recorded):
    monkeypatch.setattr(timing._FirstDataEvent, "feed", lambda *args: (_ for _ in ()).throw(ValueError("private")))
    source = ScriptedStream([(0, b"data: answer\n\n")])
    with timing.TimedHttpClient(transport=httpx.MockTransport(lambda request: httpx.Response(200,
            headers={"content-type": "text/event-stream"}, stream=source))) as client:
        with span("rag.generation"):
            assert client.post("https://provider.invalid/chat").content == b"data: answer\n\n"
    request = by_name(recorded)["llm.http.request"]
    assert request.attributes["llm.http.event_observation_unavailable"] is True
    assert request.attributes["llm.http.response_complete"] is True and source.closed


def test_no_recording_span_keeps_original_client_behavior(recorded):
    with timing.TimedHttpClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="answer"))) as client:
        assert client.get("https://provider.invalid/chat").text == "answer"
    assert recorded.get_finished_spans() == ()


def test_proxy_and_sdk_client_defaults_are_preserved(monkeypatch):
    monkeypatch.setenv("OPENAI_PROXY", "http://127.0.0.1:12345")
    options = timing.timing_client_options()
    try:
        assert options["openai_proxy"] == ""
        with DefaultHttpxClient() as original:
            assert options["http_client"].timeout == original.timeout
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model="mock", api_key="test-key", **options)
        assert model.http_client is options["http_client"]
        assert model.http_async_client is options["http_async_client"]
    finally:
        options["http_client"].close()
        asyncio.run(options["http_async_client"].aclose())


def test_async_stream_and_cancel_release_connection(recorded):
    async def check():
        class Source(httpx.AsyncByteStream):
            closed = False
            async def __aiter__(self):
                yield b": ping\n\n"
                await asyncio.sleep(0)
                yield b"data: answer\n\n"
            async def aclose(self):
                self.closed = True
        for cancel in (False, True):
            source = Source()
            async with timing.TimedAsyncHttpClient(transport=httpx.MockTransport(lambda request: httpx.Response(200,
                    headers={"content-type": "text/event-stream"}, stream=source))) as client:
                with span("rag.generation") as parent, timing.http_timing_parent(parent):
                    response = await client.send(client.build_request("POST", "https://provider.invalid/chat"), stream=True)
                    if cancel:
                        assert await anext(response.aiter_bytes()) == b": ping\n\n"
                        await response.aclose()
                    else:
                        assert await response.aread() == b": ping\n\ndata: answer\n\n"
                    assert trace.get_current_span() is parent and source.closed
    asyncio.run(check())
    requests = [item for item in recorded.get_finished_spans() if item.name == "llm.http.request"]
    assert len(requests) == 2
    assert requests[0].attributes["llm.http.first_event_received"] is True
    assert requests[0].attributes["llm.http.response_complete"] is True
    assert requests[1].attributes["llm.http.cancelled"] is True


def test_async_cancellation_during_read_finishes_spans(recorded):
    async def check():
        entered = asyncio.Event()
        class Source(httpx.AsyncByteStream):
            closed = False
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b"unused"
            async def aclose(self):
                self.closed = True
        source = Source()
        async with timing.TimedAsyncHttpClient(transport=httpx.MockTransport(lambda request: httpx.Response(200,
                headers={"content-type": "text/event-stream"}, stream=source))) as client:
            with span("rag.generation"):
                task = asyncio.create_task(client.post("https://provider.invalid/chat"))
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert source.closed
    asyncio.run(check())
    spans = by_name(recorded)
    assert len(spans) == 5
    assert spans["llm.http.request"].attributes["llm.http.cancelled"] is True

