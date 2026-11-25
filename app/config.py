"""LLM provider configuration from environment variables."""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)  # repr=False keeps the key out of logs


def load_providers(env: Mapping[str, str] | None = None) -> dict[str, ProviderConfig]:
    """Return the complete providers from LLM_PROVIDERS, in declared order.

    Without `env`, read `.env` and the process environment. Providers missing a URL or model
    are skipped with a warning. The key is optional because local Ollama does not need one.
    """
    if env is None:
        load_dotenv()
        env = os.environ

    names = [n.strip().lower() for n in env.get("LLM_PROVIDERS", "").split(",") if n.strip()]
    providers: dict[str, ProviderConfig] = {}
    for name in dict.fromkeys(names):
        prefix = name.upper()
        base_url = env.get(f"{prefix}_BASE_URL", "").strip()
        model = env.get(f"{prefix}_MODEL", "").strip()
        if not base_url or not model:
            logger.warning("Skipping provider '%s': missing _BASE_URL or _MODEL", name)
            continue
        api_key = env.get(f"{prefix}_API_KEY", "").strip()
        providers[name] = ProviderConfig(name, base_url, model, api_key)
    return providers


def is_retrieval_only(providers: Mapping[str, ProviderConfig]) -> bool:
    """With no providers the system returns excerpts and no generated answer."""
    return not providers
