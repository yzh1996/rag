"""Provider-reported token usage; unknown usage is not estimated."""
from langchain_core.callbacks import BaseCallbackHandler

from config.settings import settings


class UsageTracker(BaseCallbackHandler):
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        messages = [getattr(generation, "message", None) for batch in response.generations for generation in batch]
        usages = [getattr(message, "usage_metadata", None) for message in messages]
        usages = [usage for usage in usages if usage]
        if not usages:
            usage = (response.llm_output or {}).get("token_usage") or {}
            usages = [{"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}]
        for usage in usages:
            self.input_tokens += int(usage.get("input_tokens", 0))
            self.output_tokens += int(usage.get("output_tokens", 0))

    @property
    def cost_usd(self):
        return (self.input_tokens * settings.LLM_INPUT_COST_PER_MILLION + self.output_tokens * settings.LLM_OUTPUT_COST_PER_MILLION) / 1_000_000
