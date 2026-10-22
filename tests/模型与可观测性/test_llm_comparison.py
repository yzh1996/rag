"""离线验证测速类：使用真实 OpenAI SDK 和模拟流，不调用外部模型。"""
import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from scripts.compare_llm_providers import LLMComparisonTest, ModelConfig, load_config, main, print_results


class MockStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for delay, payload in self.chunks:
            if delay:
                await asyncio.sleep(delay)
            yield payload.encode("utf-8")

    async def aclose(self):
        self.closed = True


def event(content=None, *, reasoning=None, finish=None, usage=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    payload = {"id": "mock", "object": "chat.completion.chunk", "created": 0,
               "model": "actual-mock-model", "choices": [] if usage else [
                   {"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        payload["usage"] = usage
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def factory(handler):
    def create(**kwargs):
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return create


def model(name="GPT"):
    return ModelConfig(name, "requested-model", "https://mock.invalid/v1", "test-secret")


def test_three_services_start_together_with_identical_messages_and_no_retries():
    async def run():
        payloads, requests, streams = [], [], []
        all_started = asyncio.Event()
        async def handler(request):
            requests.append(request)
            payloads.append(json.loads(request.content))
            if len(requests) == 3:
                all_started.set()
            # 顺序调用会在这里超时；必须三家都进入请求处理才能继续。
            await asyncio.wait_for(all_started.wait(), 1)
            stream = MockStream([(0, event("年假")), (0, event("5天\n")),
                                 (0, event(finish="stop")), (0, "data: [DONE]\n\n")])
            streams.append(stream)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        results = await LLMComparisonTest([model("混元"), model("GPT"), model("千问")],
                                          client_factory=factory(handler)).run("相同上下文", "年假几天")
        assert len(requests) == 3
        assert payloads == [payloads[0]] * 3
        assert payloads[0]["messages"] == [{"role": "system", "content": "相同上下文"},
                                            {"role": "user", "content": "年假几天"}]
        assert payloads[0]["stream_options"] == {"include_usage": True}
        assert all(request.headers["x-stainless-retry-count"] == "0" for request in requests)
        assert all(result.status == "success" and result.answer == "年假5天\n" for result in results)
        assert all(result.reported_model == "actual-mock-model" for result in results)
        assert all(stream.closed for stream in streams)
    asyncio.run(run())


def test_first_text_ignores_reasoning_empty_and_whitespace_and_usage_is_preserved():
    stream = MockStream([
        (0, event()), (0.01, event(reasoning="内部思考")), (0.01, event(" \n")),
        (0.01, event("正文")), (0.01, event("第二段")), (0, event(finish="stop")),
        (0, event(usage={"prompt_tokens": 6142, "completion_tokens": 165, "total_tokens": 6307})),
        (0, "data: [DONE]\n\n"),
    ])
    async def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    result = asyncio.run(LLMComparisonTest([model()], client_factory=factory(handler)).run("提示", "问题"))[0]
    assert result.status == "success"
    assert result.answer == " \n正文第二段" and "思考" not in result.answer
    assert result.first_chunk_seconds < result.first_token_seconds < result.total_seconds
    assert result.first_token_seconds >= 0.02
    assert result.input_tokens == 6142 and result.output_tokens == 165
    assert stream.closed


@pytest.mark.parametrize("finish,content,status", [
    ("length", "部分回答", "truncated"),
    (None, "断开的回答", "error"),
    ("content_filter", "部分回答", "error"),
    ("stop", " \n", "error"),
])
def test_truncation_disconnect_filter_and_no_body_are_not_success(finish, content, status):
    stream = MockStream([(0, event(content, reasoning="思考")), (0, event(finish=finish)),
                         (0, "data: [DONE]\n\n")])
    async def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    result = asyncio.run(LLMComparisonTest([model()], client_factory=factory(handler)).run("提示", "问题"))[0]
    assert result.status == status
    assert result.input_tokens is None and result.output_tokens is None
    if not content.strip():
        assert result.first_token_seconds is None
    assert stream.closed


def test_http_error_isolated_and_response_secrets_are_not_printed(capsys):
    count = 0
    async def handler(request):
        nonlocal count
        count += 1
        if request.url.host == "bad.invalid":
            return httpx.Response(401, json={"error": {"message": "test-secret 私密业务资料", "type": "auth"}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=MockStream([(0, event("正常回答", finish="stop")),
                                                 (0, "data: [DONE]\n\n")]))
    bad = ModelConfig("失败服务", "mock", "https://bad.invalid/v1", "test-secret")
    results = asyncio.run(LLMComparisonTest([bad, model()], client_factory=factory(handler)).run("提示", "问题"))
    assert count == 2 and [result.status for result in results] == ["error", "success"]
    assert results[0].error == "HTTP 401：密钥无效"
    print_results(results)
    output = capsys.readouterr().out
    assert "test-secret" not in output and "私密业务资料" not in output
    assert "test-secret" not in repr(bad)
    assert "未提供/未提供" in output


def test_total_deadline_closes_slow_stream_and_keeps_partial_answer():
    streams = []
    async def handler(request):
        slow = request.url.host == "slow.invalid"
        stream = MockStream([(0, event("半段")), (0.2 if slow else 0, event("完成", finish="stop")),
                             (0, "data: [DONE]\n\n")])
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    slow = ModelConfig("慢服务", "mock", "https://slow.invalid/v1", "test-secret")
    results = asyncio.run(LLMComparisonTest([slow, model()], timeout=0.03,
                                          client_factory=factory(handler)).run("提示", "问题"))
    assert [result.status for result in results] == ["error", "success"]
    assert results[0].answer == "半段" and results[0].first_token_seconds is not None
    assert "超时" in results[0].error
    assert all(stream.closed for stream in streams)


def test_usage_can_be_disabled_without_fabricating_counts():
    async def handler(request):
        assert "stream_options" not in json.loads(request.content)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=MockStream([(0, event("正文", finish="stop")), (0, "data: [DONE]\n\n")]))
    target = model()
    target.include_usage = False
    result = asyncio.run(LLMComparisonTest([target], client_factory=factory(handler)).run("提示", "问题"))[0]
    assert result.status == "success" and result.input_tokens is None and result.output_tokens is None


def sample_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"system_prompt": "提示", "question": "问题", "providers": [
        {"name": "GPT", "model": "json-model", "base_url": "https://mock.invalid/v1",
         "api_key": "json-gpt-secret"},
        {"name": "千问", "model": "qwen", "base_url": "https://mock.invalid/v1",
         "api_key": ""},
    ]}), encoding="utf-8")
    return path


def test_json_config_reaches_sdk_without_environment_overrides(tmp_path, monkeypatch):
    path = sample_config(tmp_path)
    monkeypatch.setenv("OPENAI_MODEL_NAME", "env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
    models, prompt, question = load_config(path)
    assert models[0].model == "json-model"
    assert models[0].base_url == "https://mock.invalid/v1"
    assert models[0].api_key == "json-gpt-secret"
    async def handler(request):
        assert str(request.url) == "https://mock.invalid/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer json-gpt-secret"
        payload = json.loads(request.content)
        assert payload["model"] == "json-model"
        assert payload["messages"] == [{"role": "system", "content": "提示"},
                                        {"role": "user", "content": "问题"}]
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=MockStream([(0, event("正常回答", finish="stop")),
                                                 (0, "data: [DONE]\n\n")]))
    results = asyncio.run(LLMComparisonTest(models[:1], client_factory=factory(handler)).run(prompt, question))
    assert results[0].status == "success"


def test_cli_reads_json_keys_and_never_calls_models_during_preflight(tmp_path, monkeypatch, capsys):
    path = sample_config(tmp_path)
    # 即使环境里已经有密钥，也不能替代 JSON 里未填写的密钥。
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-qwen-secret")
    async def forbidden(*args, **kwargs):
        pytest.fail("配置检查或缺少密钥时不能调用任何模型")
    monkeypatch.setattr(LLMComparisonTest, "run", forbidden)
    assert main(["--config", str(path), "--check"]) == 2
    assert main(["--config", str(path)]) == 2
    data = json.loads(path.read_text(encoding="utf-8"))
    data["providers"][1]["api_key"] = "json-qwen-secret"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert main(["--config", str(path), "--check"]) == 0
    output = capsys.readouterr()
    assert "json-gpt-secret" not in output.out and "json-qwen-secret" not in output.out
    assert "env-qwen-secret" not in output.out and "本次未发起模型调用" in output.out
    assert output.err == ""


@pytest.mark.parametrize("key", [None, 123, {}, []])
def test_json_key_must_be_text(tmp_path, key):
    path = sample_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["providers"][0]["api_key"] = key
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="api_key"):
        load_config(path)


def test_whitespace_key_is_missing(tmp_path):
    path = sample_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["providers"][0]["api_key"] = "   "
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_config(path)[0][0].api_key == ""


@pytest.mark.parametrize("change", [
    {"base_url": "https://name:secret@mock.invalid/v1"},
    {"base_url": "https://mock.invalid/v1?key=secret"},
    {"extra_body": {"messages": []}},
    {"extra_body": {"stream": False}},
])
def test_config_rejects_url_credentials_and_overridden_test_input(tmp_path, change):
    path = sample_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["providers"][0].update(change)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_total_timeout_must_be_positive_and_finite(timeout):
    with pytest.raises(ValueError):
        LLMComparisonTest([model()], timeout=timeout)
