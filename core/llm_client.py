"""
core/llm_client.py
------------------
Unified LLM caller: OpenAI primary → Gemini fallback.
Reads model names from environment. Retry on rate-limit.
"""

import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Try to import LLM clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from core.logger import log_llm_call

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Cached client singletons (avoid per-call instantiation overhead) ────────
_openai_client: "OpenAI | None" = None
_gemini_client = None


def _get_openai_client() -> "OpenAI":
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini_client


# Read model names from environment (with sensible defaults)
def _openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _gemini_models() -> list[str]:
    env_model = os.getenv("GEMINI_MODEL")
    if env_model:
        return [env_model]
    return ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-flash-001"]


# ── OpenAI call ────────────────────────────────

def _openai_call(prompt: str, temperature: float = 0.3, max_tokens: int = 600) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI not available")

    client = _get_openai_client()
    response = client.chat.completions.create(
        model=_openai_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ── Gemini call (cycles through model list) ────

def _try_gemini_cycle(prompt: str, temperature: float = 0.3, max_tokens: int = 600) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini not available")

    client = _get_gemini_client()
    models = _gemini_models()
    last_error = None

    for model_name in models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                time.sleep(5)
            continue

    err_msg = f"All Gemini models failed. Last error: {last_error}"
    if "429" in str(last_error) or "quota" in str(last_error).lower():
        err_msg += "\n\n[TIP] You have hit your Gemini API quota limit (RESOURCE_EXHAUSTED). Please wait a moment or check your Google AI Studio billing/plan."
    
    raise RuntimeError(err_msg)


# ── Public unified caller ────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def llm_call(prompt: str, temperature: float = 0.3, max_tokens: int = 600, conversation_id: str = "system") -> str:
    """
    Call an LLM. Tries OpenAI first, falls back to Gemini.
    Retries up to 3 times with exponential backoff.
    """
    response = ""
    model_used = "unknown"

    # Try OpenAI first
    if os.getenv("OPENAI_API_KEY") and OPENAI_AVAILABLE:
        try:
            model_used = _openai_model()
            response = _openai_call(prompt, temperature, max_tokens)
        except Exception:
            pass

    # Fall back to Gemini
    if not response and os.getenv("GEMINI_API_KEY") and GEMINI_AVAILABLE:
        model_used = _gemini_models()[0]
        response = _try_gemini_cycle(prompt, temperature, max_tokens)

    if response:
        log_llm_call(conversation_id, prompt, response, model_used)
        return response

    raise RuntimeError(
        "No valid LLM API key found or all models failed. "
        "Set OPENAI_API_KEY or GEMINI_API_KEY in your .env file."
    )


def llm_chat(prompt: str, temperature: float = 0.7, max_tokens: int = 700, conversation_id: str = "system") -> str:
    """Conversational response generation. Uses higher temperature than llm_call."""
    return llm_call(prompt, temperature=temperature, max_tokens=max_tokens, conversation_id=conversation_id)
