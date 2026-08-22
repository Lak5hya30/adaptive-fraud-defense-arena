"""Provider-agnostic LLM client (Anthropic default) with on-disk caching.

Design goals:
- All GenAI attacker-agent calls funnel through here.
- Every response is cached to artifacts_cache/ keyed by a hash of (model, prompt,
  system, schema). This makes the committed demo fully reproducible and lets the
  whole pipeline + web app run OFFLINE from cache when no API key is present.
- structured() enforces a JSON object response and repairs common wrapping.

Swapping providers means implementing one `_complete_raw` variant; the caching,
structured-output, and offline layers are provider-neutral.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import config


class LLMUnavailable(RuntimeError):
    """Raised when a live call is needed but no key/cache is available."""


def _cache_key(model: str, system: str, prompt: str, schema_tag: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(system.encode())
    h.update(b"\x00")
    h.update(prompt.encode())
    h.update(b"\x00")
    h.update(schema_tag.encode())
    return h.hexdigest()[:24]


class LLMClient:
    """Thin wrapper around Anthropic Messages API with a JSON disk cache."""

    def __init__(self, cache_dir: Path | None = None, model: str | None = None):
        self.cache_dir = Path(cache_dir or config.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or config.LLM_MODEL
        self._client = None  # lazy

    # -- public -----------------------------------------------------------
    def complete(self, prompt: str, system: str = "", *, max_tokens: int | None = None,
                 temperature: float = 0.7, use_cache: bool = True) -> str:
        """Return raw text completion, cached to disk."""
        key = _cache_key(self.model, system, prompt, "text")
        cache_path = self.cache_dir / f"{key}.json"
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["text"]

        text = self._complete_raw(prompt, system, max_tokens or config.LLM_MAX_TOKENS,
                                  temperature)
        cache_path.write_text(json.dumps({"text": text, "prompt": prompt,
                                          "system": system, "model": self.model},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
        return text

    def structured(self, prompt: str, system: str = "", *, schema_hint: str = "object",
                   max_tokens: int | None = None, temperature: float = 0.6,
                   use_cache: bool = True) -> Any:
        """Return a parsed JSON object. Robust to markdown fences / prose wrapping."""
        key = _cache_key(self.model, system, prompt, f"json:{schema_hint}")
        cache_path = self.cache_dir / f"{key}.json"
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["data"]

        sys_full = (system + "\n\nRespond with a single valid JSON "
                    f"{schema_hint} and nothing else. No prose, no markdown fences.")
        raw = self._complete_raw(prompt, sys_full.strip(),
                                 max_tokens or config.LLM_MAX_TOKENS, temperature)
        data = _extract_json(raw)
        cache_path.write_text(json.dumps({"data": data, "raw": raw,
                                          "prompt": prompt, "model": self.model},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    def cached_only(self, prompt: str, system: str = "", *, structured: bool = False,
                    schema_hint: str = "object") -> Any | None:
        """Return a cached result if present, else None. Never hits the network."""
        tag = f"json:{schema_hint}" if structured else "text"
        key = _cache_key(self.model, system, prompt, tag)
        cache_path = self.cache_dir / f"{key}.json"
        if not cache_path.exists():
            return None
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
        return blob["data"] if structured else blob["text"]

    # -- provider impl ----------------------------------------------------
    def _complete_raw(self, prompt: str, system: str, max_tokens: int,
                      temperature: float) -> str:
        if not config.llm_available():
            raise LLMUnavailable(
                "No ANTHROPIC_API_KEY set and result not in cache. "
                "Set the key in .env to enable live generation, or run against the "
                "committed artifact cache.")
        if self._client is None:
            import anthropic  # imported lazily so offline runs never need the dep

            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a payment-fraud red-team simulation engine.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")


def _extract_json(raw: str) -> Any:
    """Best-effort JSON extraction from an LLM response."""
    raw = raw.strip()
    # Strip markdown fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced { } or [ ] span.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = raw.find(open_c)
        end = raw.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]!r}")


# Module-level singleton for convenience.
_default_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
