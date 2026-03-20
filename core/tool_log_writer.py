"""
core/tool_log_writer.py
------------------------
Writes comprehensive, human-readable log files for every agent run.

Creates two kinds of files in logs/:
  1. logs/YYYY-MM-DD_HH-MM-SS_<conv_id>.log
       Full conversation transcript with every tool called, its full input,
       full output, and the final agent response.

  2. logs/tools/<tool_name>.log  (append mode)
       One log file per tool — every invocation ever, across all conversations.

Call write_conversation_log(conversation_id, business_name) at the end of
each conversation turn (or on demand).  It reads from the existing DB tables
(tool_calls, conversations, agent_logs) so no schema changes are needed.
"""

import os
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR      = os.path.join(_PROJECT_ROOT, "logs")
TOOLS_LOG_DIR = os.path.join(LOGS_DIR, "tools")


def _get_db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _ensure_dirs() -> None:
    os.makedirs(LOGS_DIR,      exist_ok=True)
    os.makedirs(TOOLS_LOG_DIR, exist_ok=True)


def _fmt_json(obj) -> str:
    """Pretty-print a JSON-able object; fall back to str."""
    try:
        if isinstance(obj, str):
            parsed = json.loads(obj)
            return json.dumps(parsed, indent=4, ensure_ascii=False, default=str)
        return json.dumps(obj, indent=4, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _separator(char: str = "─", width: int = 72) -> str:
    return char * width


# ─────────────────────────────────────────────────────────────────────────────
# Per-tool rolling log  (logs/tools/<tool_name>.log)
# ─────────────────────────────────────────────────────────────────────────────

def append_tool_log(
    tool_name: str,
    conversation_id: str,
    timestamp: str,
    inputs: dict | str,
    outputs: dict | str,
    activated: bool = True,
) -> None:
    """
    Append one tool invocation entry to logs/tools/<tool_name>.log.
    Called automatically from write_conversation_log but can also be called
    directly from tools.py right after each tool runs.
    """
    _ensure_dirs()
    log_path = os.path.join(TOOLS_LOG_DIR, f"{tool_name}.log")

    lines = [
        _separator("═"),
        f"TOOL       : {tool_name}",
        f"TIME       : {timestamp}",
        f"CONV_ID    : {conversation_id}",
        f"ACTIVATED  : {'YES' if activated else 'NO (skipped)'}",
        _separator(),
        "── INPUTS ──────────────────────────────────────────────────────────────",
        _fmt_json(inputs),
        "── OUTPUTS ─────────────────────────────────────────────────────────────",
        _fmt_json(outputs),
        "",
    ]

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-conversation log  (logs/<timestamp>_<conv_id_prefix>.log)
# ─────────────────────────────────────────────────────────────────────────────

def write_conversation_log(
    conversation_id: str,
    business_name: str = "",
    label: str = "",
) -> str:
    """
    Read all DB records for this conversation and write a single, complete
    human-readable log file.

    Returns the path of the written file.
    """
    _ensure_dirs()
    conn = _get_db()
    try:
        # ── 1. Conversation messages ─────────────────────────────────────────
        messages = conn.execute(
            """SELECT role, content, timestamp
               FROM conversations
               WHERE conversation_id = ?
               ORDER BY id""",
            (conversation_id,),
        ).fetchall()

        # ── 2. Tool calls ────────────────────────────────────────────────────
        tool_calls = conn.execute(
            """SELECT tool_name, inputs, outputs, activated, timestamp
               FROM tool_calls
               WHERE conversation_id = ?
               ORDER BY id""",
            (conversation_id,),
        ).fetchall()

        # ── 3. Agent logs (routing, LLM calls, critic, chunk retrieval) ──────
        agent_logs = conn.execute(
            """SELECT event_type, details, timestamp
               FROM agent_logs
               WHERE conversation_id = ?
               ORDER BY id""",
            (conversation_id,),
        ).fetchall()
    finally:
        conn.close()

    # ── Build file ────────────────────────────────────────────────────────────
    now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    conv_pfx = conversation_id[:8]
    filename = f"{now_str}_{conv_pfx}.log"
    if label:
        safe_label = label.lower().replace(" ", "_")[:30]
        filename   = f"{now_str}_{safe_label}_{conv_pfx}.log"
    filepath = os.path.join(LOGS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        # Header
        f.write(_separator("═") + "\n")
        f.write(f"  AGENT CONVERSATION LOG\n")
        f.write(f"  Business    : {business_name or '(unknown)'}\n")
        f.write(f"  Conversation: {conversation_id}\n")
        f.write(f"  Written at  : {now_str} UTC\n")
        f.write(f"  Messages    : {len(messages)}\n")
        f.write(f"  Tool calls  : {len(tool_calls)}\n")
        f.write(_separator("═") + "\n\n")

        # ── Section 1: Conversation transcript ────────────────────────────────
        f.write(_separator("─") + "\n")
        f.write("  CONVERSATION TRANSCRIPT\n")
        f.write(_separator("─") + "\n\n")

        for msg in messages:
            role      = (msg["role"] or "").upper()
            content   = msg["content"] or ""
            ts        = msg["timestamp"] or ""
            role_icon = "👤" if role == "USER" else "🤖"
            f.write(f"{role_icon} [{ts}] {role}\n")
            f.write(f"{content}\n\n")

        # ── Section 2: Tool calls — full inputs & outputs ─────────────────────
        f.write(_separator("═") + "\n")
        f.write("  TOOL CALLS — FULL DETAIL\n")
        f.write(_separator("═") + "\n\n")

        if not tool_calls:
            f.write("  (no tool calls this conversation)\n\n")
        else:
            for idx, tc in enumerate(tool_calls, 1):
                activated = bool(tc["activated"])
                f.write(_separator("─") + "\n")
                f.write(f"  TOOL #{idx}: {tc['tool_name']}\n")
                f.write(f"  Time      : {tc['timestamp']}\n")
                f.write(f"  Activated : {'YES' if activated else 'NO (skipped)'}\n")
                f.write(_separator("─") + "\n")
                f.write("  ▸ INPUTS:\n")
                f.write(_fmt_json(tc["inputs"]) + "\n\n")
                f.write("  ▸ OUTPUTS:\n")
                f.write(_fmt_json(tc["outputs"]) + "\n\n")

                # Also write to per-tool rolling log
                append_tool_log(
                    tool_name=tc["tool_name"],
                    conversation_id=conversation_id,
                    timestamp=tc["timestamp"] or now_str,
                    inputs=tc["inputs"],
                    outputs=tc["outputs"],
                    activated=activated,
                )

        # ── Section 3: Agent reasoning / LLM calls ────────────────────────────
        f.write(_separator("═") + "\n")
        f.write("  AGENT REASONING & LLM CALL LOG\n")
        f.write(_separator("═") + "\n\n")

        if not agent_logs:
            f.write("  (no agent events logged)\n\n")
        else:
            for ev in agent_logs:
                event_type = ev["event_type"] or ""
                ts         = ev["timestamp"] or ""
                raw_det    = ev["details"] or "{}"

                f.write(f"  ▸ [{ts}] {event_type}\n")
                try:
                    det = json.loads(raw_det)
                    if isinstance(det, dict):
                        for k, v in det.items():
                            val_str = str(v)
                            if len(val_str) > 300:
                                val_str = val_str[:300] + "…"
                            f.write(f"      {k}: {val_str}\n")
                    else:
                        f.write(f"      {_fmt_json(det)}\n")
                except Exception:
                    f.write(f"      {str(raw_det)[:400]}\n")
                f.write("\n")

        f.write(_separator("═") + "\n")
        f.write("  END OF LOG\n")
        f.write(_separator("═") + "\n")

    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# Live console printer (called mid-turn for real-time visibility)
# ─────────────────────────────────────────────────────────────────────────────

def print_tool_log(
    tool_name: str,
    inputs: dict,
    outputs: dict,
    activated: bool = True,
) -> None:
    """
    Print a formatted tool log block to the console during agent execution.
    Shows the full inputs and outputs so nothing is hidden.
    """
    status = "✅ ACTIVATED" if activated else "⏭  SKIPPED"
    print(f"\n  {_separator('─', 60)}")
    print(f"  🔧 TOOL: {tool_name}  [{status}]")
    print(f"  {_separator('─', 60)}")
    print("  ▸ INPUTS:")
    for line in _fmt_json(inputs).splitlines():
        print(f"    {line}")
    print("  ▸ OUTPUTS:")
    for line in _fmt_json(outputs).splitlines():
        print(f"    {line}")
    print(f"  {_separator('─', 60)}")