"""Track LLM usage stats for agentic runs."""
from __future__ import annotations

from typing import Any, Dict, Optional

from langchain.callbacks.base import BaseCallbackHandler


class UsageTracker(BaseCallbackHandler):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.calls += 1
        usage = None
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage")
        if not usage:
            return
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self.total_tokens += int(usage.get("total_tokens", 0))

    def summary(self) -> Dict[str, Optional[int]]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
