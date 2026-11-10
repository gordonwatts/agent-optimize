"""Explicit provider configuration; credentials are never part of its identity."""

from dataclasses import dataclass
import math
import os
from pathlib import Path

from .generator import IDENTITY

KEY_ENV = "AGENT_OPTIMIZE_OAI_API_KEY"
MODELS = ("gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna")


def contains_credential(value, key):
    """Check decoded values as well as JSON escapes before any persistence."""
    if not key:
        return False
    if isinstance(value, str):
        return key in value
    if isinstance(value, dict):
        return any(contains_credential(part, key) for pair in value.items() for part in pair)
    if isinstance(value, (list, tuple)):
        return any(contains_credential(part, key) for part in value)
    return False


@dataclass(frozen=True)
class ProviderSettings:
    kind: str = "mock"
    model: str = "gpt-6.1-sol"
    timeout: float = 30
    max_output_tokens: int = 4096

    def __post_init__(self):
        if self.kind not in ("mock", "openai") or self.model not in MODELS:
            raise ValueError("unsupported provider or model")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("provider timeout must be finite and positive")
        if type(self.max_output_tokens) is not int or not 1 <= self.max_output_tokens <= 128000:
            raise ValueError("max output tokens must be an integer between 1 and 128000")

    @property
    def identity(self):
        if self.kind == "mock":
            return dict(IDENTITY)
        return {"kind": "openai", "identity": "openai-responses-structured-v1", "model": self.model,
                "settings": {"timeout_seconds": self.timeout, "max_output_tokens": self.max_output_tokens,
                             "max_retries": 0, "store": False, "service_tier": "default", "cache_write_ttl": "30m (API default)"}}


def _dotenv_key(path):
    try:
        # Small, literal-only dotenv reader: no interpolation or environment mutation.
        with path.open(encoding="utf-8-sig") as stream:
            content = stream.read(65537)
        if len(content) > 65536:
            raise ValueError("credential file is too large")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError):
        raise ValueError("credential file could not be read") from None
    found = None
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == KEY_ENV:
            value = value.strip()
            if value[:1] in ("'", '"'):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ValueError("credential file has an invalid quoted value")
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].rstrip()
            found = value or None
    return found


def _project_root():
    current = Path.cwd()
    return next((parent for parent in (current, *current.parents)
                 if (parent / ".git").exists() or (parent / "pyproject.toml").is_file()), current)


def resolve_api_key(explicit=None, *, environment=None, project_root=None, home=None):
    """CLI > named environment > project .env > home .env; no OPENAI fallback."""
    environment = os.environ if environment is None else environment
    key = explicit or environment.get(KEY_ENV)
    if not key:
        key = _dotenv_key(Path(project_root or _project_root()) / ".env")
    if not key:
        key = _dotenv_key(Path(home or Path.home()) / ".env")
    if not key or not key.strip():
        raise ValueError("OpenAI provider requires AGENT_OPTIMIZE_OAI_API_KEY in the environment or .env, or --api-key")
    return key.strip()
