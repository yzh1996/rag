"""模型 HTTP 请求的基础计时，不记录请求或响应正文。"""
import asyncio
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar

import httpx
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient
from opentelemetry import trace

_http_parent = ContextVar("llm_http_parent", default=None)


@contextmanager
def http_timing_parent(parent):
    """流式请求跨越首段等待和输出阶段，统一挂到整个生成阶段。"""
    token = _http_parent.set(trace.set_span_in_context(parent))
    try:
        yield
    finally:
        _http_parent.reset(token)


def http_timing_chunks(chunks, parent):
    # LangChain 可能在复制的上下文中关闭生成器；绑定只覆盖 next，不跨越 yield。
    while True:
        with http_timing_parent(parent):
            try:
                item = next(chunks)
            except StopIteration:
                return
        yield item


class _FirstDataEvent:
    """只识别 SSE 的 data 行和事件边界，不保留事件内容。"""

    def __init__(self):
        self.prefix = bytearray()
        self.line_size = 0
        self.has_data = False
        self.skip_lf = False

    def feed(self, data):
        for byte in data:
            if self.skip_lf:
                self.skip_lf = False
                if byte == 10:
                    continue
            if byte in (10, 13):
                if self.line_size == 0 and self.has_data:
                    return True
                if self.prefix == b"data:" or (self.line_size == 4 and self.prefix == b"data"):
                    self.has_data = True
                self.prefix.clear()
                self.line_size = 0
                self.skip_lf = byte == 13
            else:
                if len(self.prefix) < 5:
                    self.prefix.append(byte)
                self.line_size += 1
        return False


class _HttpTiming:
    def __init__(self, request):
        self.tracer = trace.get_tracer("rag")
        self.started_at = time.perf_counter()
        self.request = self.tracer.start_span("llm.http.request", context=_http_parent.get())
        self.context = trace.set_span_in_context(self.request)
        self.headers_wait = self.tracer.start_span("llm.http.wait_headers", context=self.context)
        self.body = None
        self.event_wait = None
        self.parser = None
        self.decoder = None
        self.finished = False
        self.request.set_attribute("http.request.method", request.method)
        self.request.add_event("request.started")

    def received_headers(self, response):
        self.headers_at = time.perf_counter()
        self.request.set_attribute("http.response.status_code", response.status_code)
        self.request.set_attribute("llm.http.headers_ms", round((self.headers_at - self.started_at) * 1000, 2))
        self.request.add_event("response.headers_received")
        self.headers_wait.end()
        self.headers_wait = None
        if response.status_code >= 400:
            self.request.set_status(trace.Status(trace.StatusCode.ERROR))
        self.body = self.tracer.start_span("llm.http.read_body", context=self.context)
        if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() == "text/event-stream":
            self.request.set_attribute("llm.http.first_event_received", False)
            self.event_wait = self.tracer.start_span("llm.http.wait_first_event", context=trace.set_span_in_context(self.body))
            self.parser = _FirstDataEvent()
            try:
                # 独立的观察解码器支持压缩响应；不触碰真实响应的解码状态。
                preview = httpx.Response(200, headers={"content-encoding": response.headers.get("content-encoding", "")})
                self.decoder = preview._get_content_decoder()
            except Exception:
                self.request.set_attribute("llm.http.event_observation_unavailable", True)
                self.parser = None

    def observe(self, data, final=False):
        if self.parser is None or self.finished:
            return
        try:
            decoded = self.decoder.flush() if final else self.decoder.decode(data)
            if not self.parser.feed(decoded):
                return
        except Exception:
            # 观察失败只影响计时，响应仍交给原客户端处理。
            self.request.set_attribute("llm.http.event_observation_unavailable", True)
            self.parser = None
            return
        now = time.perf_counter()
        self.request.set_attribute("llm.http.first_event_received", True)
        self.request.set_attribute("llm.http.first_event_ms", round((now - self.started_at) * 1000, 2))
        self.request.set_attribute("llm.http.headers_to_first_event_ms", round((now - self.headers_at) * 1000, 2))
        self.request.add_event("response.first_data_event_received")
        self.event_wait.end()
        self.event_wait = None
        self.parser = None

    def finish(self, complete=False, error=None):
        if self.finished:
            return
        self.finished = True
        active = [item for item in (self.headers_wait, self.event_wait, self.body, self.request) if item is not None]
        if error is not None:
            for item in active:
                if isinstance(error, Exception):
                    item.set_attribute("error.type", type(error).__name__)
                    item.set_status(trace.Status(trace.StatusCode.ERROR))
                else:
                    item.set_attribute("llm.http.cancelled", True)
        elif not complete:
            self.request.set_attribute("llm.http.cancelled", True)
        self.request.set_attribute("llm.http.response_complete", complete)
        self.request.set_attribute("llm.http.total_ms", round((time.perf_counter() - self.started_at) * 1000, 2))
        self.request.add_event("request.finished")
        for item in active:
            item.end()


class _TimedSyncStream(httpx.SyncByteStream):
    def __init__(self, stream, timing):
        self.stream = stream
        self.timing = timing
        self.complete = False

    def __iter__(self):
        try:
            for data in self.stream:
                self.timing.observe(data)
                yield data
            self.timing.observe(b"", final=True)
            self.complete = True
        except BaseException as exc:
            self.timing.finish(error=exc)
            raise

    def close(self):
        try:
            self.stream.close()
        except BaseException as exc:
            self.timing.finish(error=exc)
            raise
        finally:
            self.timing.finish(complete=self.complete)


class _TimedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, stream, timing):
        self.stream = stream
        self.timing = timing
        self.complete = False

    async def __aiter__(self):
        try:
            async for data in self.stream:
                self.timing.observe(data)
                yield data
            self.timing.observe(b"", final=True)
            self.complete = True
        except BaseException as exc:
            self.timing.finish(error=exc)
            raise

    async def aclose(self):
        try:
            await self.stream.aclose()
        except BaseException as exc:
            self.timing.finish(error=exc)
            raise
        finally:
            self.timing.finish(complete=self.complete)


class TimedHttpClient(DefaultHttpxClient):
    def __del__(self):
        # 与 SDK 默认客户端一样，在模型客户端被释放时关闭连接池。
        try:
            if not self.is_closed:
                self.close()
        except Exception:
            pass

    def send(self, request, *, stream=False, **kwargs):
        if not trace.get_current_span().is_recording():
            return super().send(request, stream=stream, **kwargs)
        timing = _HttpTiming(request)
        response = None
        try:
            response = super().send(request, stream=True, **kwargs)
            timing.received_headers(response)
            if response.is_closed:
                timing.observe(response.content)
                timing.observe(b"", final=True)
                timing.finish(complete=True)
            else:
                response.stream = _TimedSyncStream(response.stream, timing)
                if not stream:
                    response.read()
            return response
        except BaseException as exc:
            timing.finish(error=exc)
            if response is not None:
                response.close()
            raise


class TimedAsyncHttpClient(DefaultAsyncHttpxClient):
    def __del__(self):
        try:
            if not self.is_closed:
                asyncio.get_running_loop().create_task(self.aclose())
        except Exception:
            pass

    async def send(self, request, *, stream=False, **kwargs):
        if not trace.get_current_span().is_recording():
            return await super().send(request, stream=stream, **kwargs)
        timing = _HttpTiming(request)
        response = None
        try:
            response = await super().send(request, stream=True, **kwargs)
            timing.received_headers(response)
            if response.is_closed:
                timing.observe(response.content)
                timing.observe(b"", final=True)
                timing.finish(complete=True)
            else:
                response.stream = _TimedAsyncStream(response.stream, timing)
                if not stream:
                    await response.aread()
            return response
        except BaseException as exc:
            timing.finish(error=exc)
            if response is not None:
                await response.aclose()
            raise


def timing_client_options():
    # 保留 LangChain 对 OPENAI_PROXY 的支持，以及 SDK 原来的超时和连接数默认值。
    proxy = os.environ.get("OPENAI_PROXY") or None
    return {
        "http_client": TimedHttpClient(proxy=proxy),
        "http_async_client": TimedAsyncHttpClient(proxy=proxy),
        "openai_proxy": "",
    }
