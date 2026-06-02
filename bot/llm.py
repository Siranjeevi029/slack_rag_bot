"""Gemini completion with multi-key rotation and overload handling.

- TPM/RPM rate-limit on a key  -> rotate to next key immediately.
- 503 "overloaded / high demand" / transient (timeout, conn drop) on a key ->
  retry same key up to GEMINI_503_RETRIES times (with backoff), then rotate.
- Every key rate-limited in one full pass -> wait ALL_KEYS_TPM_WAIT seconds.

complete()    -> returns text.
complete_ex() -> returns (text, usage) where usage = {key_index, prompt_tokens,
                 completion_tokens, total_tokens} for logging.
"""
import time

import litellm

from bot import config

litellm.drop_params = True


def _is_transient(e: Exception) -> bool:
    """True for retryable errors: Gemini 503 overload, connection drops, timeouts."""
    s = str(e).lower()
    code = getattr(e, "status_code", None)
    if code in (500, 502, 503, 504):
        return True
    if isinstance(e, (litellm.APIConnectionError, litellm.Timeout, litellm.InternalServerError)):
        return True
    return any(t in s for t in (
        "overloaded", "high demand", "unavailable", "503",
        "disconnected", "connection", "timed out", "timeout",
    ))


def _try_key(messages: list, key: str, temperature: float, idx: int):
    """Try one key. Return (text, usage_dict) on success, or None to rotate."""
    for attempt in range(config.GEMINI_503_RETRIES):
        try:
            resp = litellm.completion(
                model=config.RAG_MODEL,
                messages=messages,
                temperature=temperature,
                api_key=key,
            )
            u = getattr(resp, "usage", None)
            usage = {
                "key_index": idx,
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None),
                "total_tokens": getattr(u, "total_tokens", None),
            }
            return resp.choices[0].message.content, usage
        except (litellm.RateLimitError, litellm.BadRequestError) as e:
            if "quota" in str(e).lower() or "429" in str(e) or "rate" in str(e).lower():
                print(f"  key #{idx} hit TPM/RPM, rotating...")
                return None
            raise
        except Exception as e:  # noqa: BLE001
            if not _is_transient(e):
                raise
            if attempt < config.GEMINI_503_RETRIES - 1:
                wait = 2 * (attempt + 1)
                print(f"  key #{idx} transient error ({type(e).__name__}), retry "
                      f"{attempt + 1}/{config.GEMINI_503_RETRIES} in {wait}s")
                time.sleep(wait)
                continue
            print(f"  key #{idx} still failing after {config.GEMINI_503_RETRIES} tries, rotating...")
            return None
    return None


def complete_ex(prompt: str, temperature: float = 0, max_all_key_waits: int = 5):
    """Run a completion; return (text, usage) with key_index + token counts."""
    keys = config.GEMINI_KEYS
    if not keys:
        raise RuntimeError("No Gemini keys found")

    messages = [{"role": "user", "content": prompt}]
    waits = 0
    while True:
        for idx, key in enumerate(keys):
            result = _try_key(messages, key, temperature, idx)
            if result is not None:
                return result  # (text, usage)
        waits += 1
        if waits > max_all_key_waits:
            raise RuntimeError(f"All Gemini keys exhausted after {max_all_key_waits} waits")
        print(f"All {len(keys)} Gemini keys hit limit — waiting {config.ALL_KEYS_TPM_WAIT}s...")
        time.sleep(config.ALL_KEYS_TPM_WAIT)


def complete(prompt: str, temperature: float = 0, max_all_key_waits: int = 5) -> str:
    text, _ = complete_ex(prompt, temperature, max_all_key_waits)
    return text
