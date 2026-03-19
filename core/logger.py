"""
core/logger.py
--------------
Centralised SQLite logging for every agent event.

All writes use WAL mode + timeout=30 to prevent lock errors.
Functions are additive: new log types added without breaking existing calls.

Tables used:
  conversations  -- user / assistant messages
  tool_calls     -- every tool invoked + inputs/outputs
  agent_logs     -- routing decisions, chunk retrieval, critic events
  llm_call_logs  -- every LLM call with prompt snippet + response snippet
"""

import sqlite3
import json
import os
from datetime import datetime, timezone


def _get_db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Conversation messages
# ─────────────────────────────────────────────────────────────────────────────

def log_message(
    conversation_id: str,
    role: str,
    content: str,
    business_name: str = "",
) -> None:
    """Log a single conversation turn (user or assistant)."""
    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO conversations
               (conversation_id, business_name, role, content, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (
                conversation_id,
                business_name,
                role,
                content,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception as exc:
        print(f"[LOGGER] log_message error: {exc}")
    finally:
        conn.close()


def get_conversation_history(
    conversation_id: str,
    limit: int = 20,
    business_name: str = "",
) -> list[dict]:
    """
    Return the last `limit` messages for a conversation, oldest-first.
    Loaded from DB (conversations table) each turn — this is the agent's memory.
    """
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT role, content, timestamp
               FROM conversations
               WHERE conversation_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (conversation_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    except Exception as exc:
        print(f"[LOGGER] get_conversation_history error: {exc}")
        return []
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tool calls
# ─────────────────────────────────────────────────────────────────────────────

def log_tool_call(
    conversation_id: str,
    tool_name: str,
    inputs: dict,
    outputs: dict,
    activated: bool = True,
) -> None:
    """Log one tool invocation with its full inputs and outputs."""
    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO tool_calls
               (conversation_id, tool_name, inputs, outputs, activated, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                conversation_id,
                tool_name,
                json.dumps(inputs,  default=str, ensure_ascii=False),
                json.dumps(outputs, default=str, ensure_ascii=False),
                1 if activated else 0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception as exc:
        print(f"[LOGGER] log_tool_call error: {exc}")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Chunk retrieval
# ─────────────────────────────────────────────────────────────────────────────

def log_chunk_retrieval(
    conversation_id: str,
    query: str,
    chunks_found: int,
    chunk_ids: list | None = None,
) -> None:
    """Log how many chunks were retrieved and which ones, for traceability."""
    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO agent_logs
               (conversation_id, event_type, details, timestamp)
               VALUES (?, ?, ?, ?)""",
            (
                conversation_id,
                "chunk_retrieval",
                json.dumps(
                    {"query": query, "chunks_found": chunks_found, "chunk_ids": chunk_ids or []},
                    ensure_ascii=False,
                ),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception as exc:
        print(f"[LOGGER] log_chunk_retrieval error: {exc}")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Generic agent events
# ─────────────────────────────────────────────────────────────────────────────

def log_agent_event(
    conversation_id: str,
    event_type: str,
    details: dict | str,
) -> None:
    """Log any agent event (routing, critic, escalation, etc.)."""
    conn = _get_db()
    try:
        if isinstance(details, dict):
            details = json.dumps(details, default=str, ensure_ascii=False)
        conn.execute(
            """INSERT INTO agent_logs
               (conversation_id, event_type, details, timestamp)
               VALUES (?, ?, ?, ?)""",
            (
                conversation_id,
                event_type,
                details,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception as exc:
        print(f"[LOGGER] log_agent_event error: {exc}")
    finally:
        conn.close()


def log_agent_reasoning(
    conversation_id: str,
    stage: str,
    details: dict,
) -> None:
    """Log a reasoning step (topic extraction, routing decision, critic output)."""
    log_agent_event(conversation_id, f"reasoning:{stage}", details)


# ─────────────────────────────────────────────────────────────────────────────
# LLM call tracing
# ─────────────────────────────────────────────────────────────────────────────

def log_llm_call(
    conversation_id: str,
    call_type: str,
    prompt_snippet: str,
    response_snippet: str,
    model: str = "",
) -> None:
    """
    Log every LLM invocation for full traceability.
    Stores first 800 chars of prompt + first 800 chars of response.
    Table: agent_logs  (event_type = "llm_call:{call_type}")
    """
    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO agent_logs
               (conversation_id, event_type, details, timestamp)
               VALUES (?, ?, ?, ?)""",
            (
                conversation_id,
                f"llm_call:{call_type}",
                json.dumps(
                    {
                        "model":    model,
                        "prompt":   prompt_snippet[:800],
                        "response": response_snippet[:800],
                    },
                    ensure_ascii=False,
                ),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception as exc:
        print(f"[LOGGER] log_llm_call error: {exc}")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# DB write audit
# ─────────────────────────────────────────────────────────────────────────────

def log_db_write(
    conversation_id: str,
    table: str,
    operation: str,
    record_id: int | None,
    details: dict | None = None,
) -> None:
    """Audit log for every database write operation (INSERT / UPDATE / DELETE)."""
    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO agent_logs
               (conversation_id, event_type, details, timestamp)
               VALUES (?, ?, ?, ?)""",
            (
                conversation_id,
                f"db_write:{table}:{operation}",
                json.dumps(
                    {"record_id": record_id, **(details or {})},
                    default=str,
                    ensure_ascii=False,
                ),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception as exc:
        print(f"[LOGGER] log_db_write error: {exc}")
    finally:
        conn.close()