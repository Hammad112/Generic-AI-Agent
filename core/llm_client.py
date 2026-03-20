"""
core/llm_client.py
------------------
Unified LLM caller supporting:
  1. llm_call()           — plain text generation (OpenAI → Gemini fallback)
  2. llm_with_tools()     — native function calling (OpenAI → Gemini fallback)
                            Returns (response_text, tool_calls_list)

Function calling format (tool_calls_list items):
    {"name": "book_appointment", "arguments": {"date": "monday", "time": "10am", ...}}
"""

import os
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from core.logger import log_llm_call

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Cached clients ──────────────────────────────────────────────────────────
_openai_client = None
_gemini_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini_client


def _openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _gemini_models() -> list[str]:
    env = os.getenv("GEMINI_MODEL")
    if env:
        return [env]
    return ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-flash-001"]


# ─────────────────────────────────────────────────────────────────────────────
# Plain text generation  (unchanged API)
# ─────────────────────────────────────────────────────────────────────────────

def _openai_call(prompt: str, temperature: float = 0.3, max_tokens: int = 600) -> str:
    if not os.getenv("OPENAI_API_KEY") or not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI not available")
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=_openai_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _try_gemini_cycle(prompt: str, temperature: float = 0.3, max_tokens: int = 600) -> str:
    if not os.getenv("GEMINI_API_KEY") or not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini not available")
    client = _get_gemini_client()
    last_error = None
    for model_name in _gemini_models():
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": temperature, "max_output_tokens": max_tokens},
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                time.sleep(5)
    raise RuntimeError(f"All Gemini models failed. Last: {last_error}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def llm_call(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 600,
    conversation_id: str = "system",
) -> str:
    response = ""
    model_used = "unknown"

    if os.getenv("OPENAI_API_KEY") and OPENAI_AVAILABLE:
        try:
            model_used = _openai_model()
            response = _openai_call(prompt, temperature, max_tokens)
        except Exception:
            pass

    if not response and os.getenv("GEMINI_API_KEY") and GEMINI_AVAILABLE:
        model_used = _gemini_models()[0]
        response = _try_gemini_cycle(prompt, temperature, max_tokens)

    if response:
        log_llm_call(conversation_id, "text_generation", prompt[:500], response[:500], model_used)
        return response

    raise RuntimeError(
        "No LLM API key found or all models failed. "
        "Set OPENAI_API_KEY or GEMINI_API_KEY in .env"
    )


def llm_chat(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 700,
    conversation_id: str = "system",
) -> str:
    return llm_call(prompt, temperature=temperature, max_tokens=max_tokens, conversation_id=conversation_id)


# ─────────────────────────────────────────────────────────────────────────────
# Function calling  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def llm_with_tools(
    messages: list[dict],
    tools: list[dict],      # OpenAI-format tool list
    conversation_id: str = "system",
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> tuple[str, list[dict]]:
    """
    Call LLM with function calling support.

    Args:
        messages: Full message history in OpenAI format:
                  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        tools:    OpenAI-format tool definitions (from tool_schemas.OPENAI_TOOLS)
        conversation_id: For logging
        temperature / max_tokens: Generation params

    Returns:
        (response_text, tool_calls)
        tool_calls: list of {"name": str, "arguments": dict}
        If no tool was called: tool_calls = []
        If tool was called: response_text = "" (will be filled after tool execution)
    """
    # ── Try OpenAI function calling ─────────────────────────────────────────
    if os.getenv("OPENAI_API_KEY") and OPENAI_AVAILABLE:
        try:
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model=_openai_model(),
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            msg = resp.choices[0].message

            # Tool call requested
            if msg.tool_calls:
                calls = []
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    calls.append({"name": tc.function.name, "arguments": args, "id": tc.id})
                log_llm_call(conversation_id, "[function_call]",
                             f"Tools called: {[c['name'] for c in calls]}",
                             json.dumps(calls, default=str)[:500],
                             _openai_model())
                return "", calls

            # Plain text response
            text = (msg.content or "").strip()
            log_llm_call(conversation_id, "[chat]",
                         "[messages → text]", text[:500], _openai_model())
            return text, []

        except Exception as exc:
            print(f"  [LLM] OpenAI function calling failed ({exc}), trying Gemini...")

    # ── Try Gemini function calling ──────────────────────────────────────────
    if os.getenv("GEMINI_API_KEY") and GEMINI_AVAILABLE:
        try:
            return _gemini_with_tools(messages, tools, conversation_id, temperature, max_tokens)
        except Exception as exc:
            print(f"  [LLM] Gemini function calling failed ({exc}), falling back to text-only...")

    # ── Last resort: plain text with routing prompt ─────────────────────────
    # Extract the last user message and use plain llm_call
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    # Build a minimal routing prompt
    tool_names = [t["function"]["name"] for t in tools[:20]]
    system_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
    fallback_prompt = (
        f"{system_msg}\n\n"
        f"Customer message: {last_user}\n\n"
        f"Respond helpfully. Available capabilities: {', '.join(tool_names)}"
    )
    text = llm_call(fallback_prompt, temperature=temperature, max_tokens=max_tokens, conversation_id=conversation_id)
    return text, []


def _gemini_with_tools(
    messages: list[dict],
    openai_tools: list[dict],
    conversation_id: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, list[dict]]:
    """Gemini function calling using google-genai SDK."""
    client = _get_gemini_client()

    # Convert OpenAI messages → Gemini contents
    # Gemini uses role "user" and "model" (not "assistant")
    system_instruction = ""
    gemini_contents = []

    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_instruction = content
            continue
        elif role == "assistant":
            role = "model"
        elif role == "tool":
            # Tool result — add as user message with result
            role = "user"
            content = f"[Tool result for {msg.get('name', 'tool')}]: {content}"

        if content:
            gemini_contents.append({"role": role, "parts": [{"text": content}]})

    if not gemini_contents:
        raise RuntimeError("No messages to send to Gemini")

    # Convert OpenAI tool schemas → Gemini function declarations
    function_declarations = []
    for tool in openai_tools:
        fn = tool.get("function", {})
        params = fn.get("parameters", {})
        props  = params.get("properties", {})
        gemini_props = {}
        for k, v in props.items():
            gp = {"type": v.get("type", "string").upper(), "description": v.get("description", "")}
            if "enum" in v:
                gp["enum"] = v["enum"]
            gemini_props[k] = gp

        decl = {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
        }
        if gemini_props:
            decl["parameters"] = {
                "type": "OBJECT",
                "properties": gemini_props,
            }
            required = params.get("required", [])
            if required:
                decl["parameters"]["required"] = required
        function_declarations.append(decl)

    config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if system_instruction:
        config["system_instruction"] = system_instruction

    last_error = None
    for model_name in _gemini_models():
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=gemini_contents,
                tools=[{"function_declarations": function_declarations}],
                config=config,
            )

            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                continue

            # Check for function calls in the response parts
            tool_calls = []
            text_parts = []

            for part in (candidate.content.parts if candidate.content else []):
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = {}
                    if fc.args:
                        # Gemini returns args as a Struct/dict-like object
                        try:
                            args = dict(fc.args)
                        except Exception:
                            args = {}
                    tool_calls.append({
                        "name": fc.name,
                        "arguments": args,
                        "id": f"gemini_{fc.name}",
                    })
                elif hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

            if tool_calls:
                log_llm_call(conversation_id, "[gemini_function_call]",
                             f"Tools called: {[tc['name'] for tc in tool_calls]}",
                             json.dumps(tool_calls, default=str)[:500], model_name)
                return "", tool_calls

            text = " ".join(text_parts).strip()
            if text:
                log_llm_call(conversation_id, "[gemini_chat]",
                             "[messages → text]", text[:500], model_name)
                return text, []

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str:
                time.sleep(5)
            continue

    raise RuntimeError(f"Gemini function calling failed. Last: {last_error}")