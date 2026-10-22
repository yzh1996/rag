"""单轮并行比较三个模型 API 的速度和回答，不经过项目检索或修改在线配置。"""
import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

DEFAULT_CONFIG = Path(__file__).with_name("llm_compare.example.json")


@dataclass
class ModelConfig:
    """一个服务的配置；密钥不出现在对象打印结果中。"""
    name: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    include_usage: bool = True
    extra_body: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    name: str
    requested_model: str
    reported_model: str | None = None
    status: str = "error"
    first_chunk_seconds: float | None = None
    first_token_seconds: float | None = None
    total_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    answer: str = ""
    error: str | None = None


class LLMComparisonTest:
    """相同消息同时发给各服务，单个服务失败不影响其他服务返回结果。"""

    def __init__(self, models: list[ModelConfig], timeout: float = 60,
                 client_factory: Callable = AsyncOpenAI):
        if not models or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("至少配置一个模型，且超时时间必须是大于零的有限值")
        self.models = models
        self.timeout = timeout
        self.client_factory = client_factory

    async def run(self, system_prompt: str, question: str) -> list[ComparisonResult]:
        if not system_prompt.strip() or not question.strip():
            raise ValueError("系统提示词和问题不能为空")
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}]
        # 同一轮并行发起，尽量减少不同测试时段造成的服务负载差异。
        return await asyncio.gather(*(self._test_one(model, messages) for model in self.models))

    async def _test_one(self, model: ModelConfig, messages: list[dict]) -> ComparisonResult:
        result = ComparisonResult(model.name, model.model)
        if not model.api_key:
            result.status, result.error = "not_configured", "未配置 API 密钥"
            return result
        started = None
        try:
            # 关闭 SDK 自动重试，避免一次失败被隐式重试后混入测速结果。
            async with self.client_factory(api_key=model.api_key, base_url=model.base_url,
                                           timeout=self.timeout, max_retries=0) as client:
                started = time.perf_counter()
                # 总时限也覆盖持续发送心跳的服务，避免流一直不结束。
                await asyncio.wait_for(self._read_stream(client, model, messages, result, started),
                                       timeout=self.timeout)
        except (asyncio.TimeoutError, APITimeoutError):
            result.error = "请求超时；已有正文属于未完成回答"
        except APIStatusError as exc:
            # 不打印供应商原始错误正文，它可能回显密钥、请求内容等数据。
            explanation = {401: "密钥无效", 403: "没有访问权限", 429: "限流或额度不足"}
            result.error = f"HTTP {exc.status_code}：{explanation.get(exc.status_code, '服务请求失败')}"
        except APIConnectionError:
            result.error = "连接失败，请检查服务地址和网络"
        except Exception as exc:
            result.error = f"调用失败（{type(exc).__name__}）"
        finally:
            if started is not None:
                result.total_seconds = time.perf_counter() - started
            if result.error:
                result.status = "error"
        return result

    @staticmethod
    async def _read_stream(client, model, messages, result, started):
        options = {"stream_options": {"include_usage": True}} if model.include_usage else {}
        if model.extra_body:
            options["extra_body"] = model.extra_body
        finish_reason = None
        async with await client.chat.completions.create(
                model=model.model, messages=messages, stream=True, **options) as stream:
            async for chunk in stream:
                elapsed = time.perf_counter() - started
                if result.first_chunk_seconds is None:
                    result.first_chunk_seconds = elapsed
                if chunk.model:
                    result.reported_model = chunk.model
                if chunk.usage is not None:
                    # 未提供用量时保留 None，不虚构零 Token。
                    result.input_tokens = chunk.usage.prompt_tokens
                    result.output_tokens = chunk.usage.completion_tokens
                for choice in chunk.choices:
                    if choice.index != 0:
                        continue
                    text = choice.delta.content or ""
                    result.answer += text
                    # role、用量、思考内容和前导空白均不算第一段可见正文。
                    if result.first_token_seconds is None and text.strip():
                        result.first_token_seconds = elapsed
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
        if not result.answer.strip():
            result.error = "服务没有返回非空答案正文"
        elif finish_reason == "length":
            result.status = "truncated"
            result.error = None
        elif finish_reason != "stop":
            result.error = "回答未正常完成，请检查连接或服务返回的结束标记"
        else:
            result.status = "success"


def load_config(path: Path) -> tuple[list[ModelConfig], str, str]:
    """直接读取 JSON 的模型、服务地址和密钥，不使用环境变量覆盖。"""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    models = []
    for item in payload["providers"]:
        model = item["model"]
        base_url = item["base_url"]
        api_key = item.get("api_key", "")
        if not isinstance(api_key, str):
            raise ValueError("api_key 必须是字符串，未填写时使用空字符串")
        api_key = api_key.strip()
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError("服务地址必须是有效的 HTTP 或 HTTPS 地址")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("服务地址不能携带凭据或查询参数，请通过 API 密钥配置认证")
        extra = item.get("extra_body", {})
        if not isinstance(extra, dict) or {"model", "messages", "stream", "stream_options", "n"} & extra.keys():
            raise ValueError("额外参数不能覆盖模型、消息、流式开关或回答数量")
        models.append(ModelConfig(item["name"], model, base_url,
                                  api_key,
                                  item.get("include_usage", True), extra))
    if not models or len({model.name for model in models}) != len(models):
        raise ValueError("服务列表不能为空，服务名称不能重复")
    return models, payload["system_prompt"], payload["question"]


def print_results(results: list[ComparisonResult]) -> None:
    def duration(value):
        return "--" if value is None else f"{value:.3f}s"
    def tokens(value):
        return "未提供" if value is None else str(value)
    labels = {"success": "完成", "error": "失败", "truncated": "被截断", "not_configured": "未配置"}
    print("\n服务 | 实际返回模型 | 首个流片段 | 首段正文 | 总耗时 | 输入/输出 Token | 状态")
    for result in results:
        model = result.reported_model or f"{result.requested_model}（请求值）"
        print(f"{result.name} | {model} | {duration(result.first_chunk_seconds)} | "
              f"{duration(result.first_token_seconds)} | {duration(result.total_seconds)} | "
              f"{tokens(result.input_tokens)}/{tokens(result.output_tokens)} | {labels[result.status]}")
    for result in results:
        print(f"\n=== {result.name}：{labels[result.status]} ===")
        if result.error:
            print(result.error)
        if result.status == "truncated":
            print("达到输出长度限制，以下不是完整回答。")
        elif result.status != "success" and result.answer:
            print("以下为未完成的部分回答，仅供排查。")
        if result.answer:
            print(result.answer)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="单轮并行比较元宝对应模型 API、GPT-6 和千问")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prompt-file", type=Path, help="完整系统提示词和上下文的 UTF-8 文本文件")
    parser.add_argument("--question", help="覆盖样例中的用户问题")
    parser.add_argument("--timeout", type=float, default=60, help="每个服务的总超时秒数")
    parser.add_argument("--check", action="store_true", help="仅检查配置，不调用模型")
    args = parser.parse_args(argv)
    try:
        # 三项配置全部来自 JSON；不读取项目 .env，也不导入项目 Settings。
        models, prompt, question = load_config(args.config)
        if args.prompt_file:
            prompt = args.prompt_file.read_text(encoding="utf-8-sig")
        if args.question is not None:
            question = args.question
        benchmark = LLMComparisonTest(models, args.timeout)
        if not prompt.strip() or not question.strip():
            raise ValueError("系统提示词和问题不能为空")
    except (OSError, ValueError, KeyError, TypeError):
        print("配置或输入无效：请检查文件路径、UTF-8/JSON 格式、必填字段、服务地址和超时时间。")
        return 2
    for model in models:
        print(f"{model.name}：模型={model.model}，服务={urlsplit(model.base_url).hostname}，"
              f"密钥={'已配置' if model.api_key else '未配置'}")
    missing = any(not model.api_key for model in models)
    if args.check:
        print("配置检查结束，没有调用任何模型。")
        return 2 if missing else 0
    if missing:
        print("请先补齐所有服务的密钥，本次未发起模型调用。")
        return 2
    print("将把同一份输入发送给以上服务，每家调用一次。元宝对应 API 不代表元宝网页版。")
    results = asyncio.run(benchmark.run(prompt, question))
    print_results(results)
    return 0 if all(result.status == "success" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
