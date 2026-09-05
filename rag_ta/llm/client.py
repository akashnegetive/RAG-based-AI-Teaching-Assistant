"""Single OpenAI client with sane timeouts and built-in retries."""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from rag_ta.config import Settings
from rag_ta.config import settings as default_settings


@lru_cache(maxsize=2)
def _make(timeout: float, max_retries: int) -> OpenAI:
    # The SDK retries 408/409/429/5xx and connection errors with exponential backoff.
    return OpenAI(timeout=timeout, max_retries=max_retries)


def get_openai(cfg: Settings = default_settings) -> OpenAI:
    return _make(cfg.openai_timeout, cfg.openai_max_retries)
