"""
core/master_log.py
-------------------
Writes ONE combined  logs/all_logs.log  containing EVERYTHING:

  ┌─ SESSION HEADER (business loaded, web searches done + links)
  ├─ TURN 1
  │    💬 Full conversation so far
  │    🔧 Tool calls this turn (full inputs + outputs)
  │    🧠 Agent reasoning (routing, LLM calls, DB writes)
  ├─ TURN 2 ...

Web search section shows:
  - How many searches were done
  - Each query + number of results + every link returned

Called automatically from agent.py node_log_turn after every turn.
Also callable standalone:  python -m core.master_log
"""

import os
import json
import sqlite3
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR      = os.path.join(_PROJECT_ROOT, "logs")
ALL_LOGS_FILE = os.path.join(LOGS_DIR, "all_logs.log")

# Track which sessions we have already written a header for (in-process memory)
_session_headers_written: set = set()


def _get_db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _line(char="─", width=80):
    return char * width


def _fmt(obj) -> str:
    try:
        if isinstance(obj, str):
            obj = json.loads(obj)
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Session header — written once per conversation_id
# ─────────────────────────────────────────────────────────────────────────────

def _write_session_header(f, business_name: str, conversation_id: str, conn) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    f.write("\n" + _line("═") + "\n")
    f.write(f"  SESSION START  |  {now}\n")
    f.write(f"  Business  : {business_name or '(unknown)'}\n")
    f.write(f"  Conv ID   : {conversation_id}\n")
    f.write(_line("═") + "\n\n")

    # ── Web search summary with queries + links ────────────────────────────
    web_articles = []
    try:
        web_articles = conn.execute(
            """SELECT topic, source, content FROM enriched_knowledge
               WHERE business_name = ? AND source LIKE 'web_search%'
               ORDER BY id""",
            (business_name,),
        ).fetchall()
    except Exception:
        pass

    # Load stored DDG detail logs (query → links) written by knowledge_enricher
    search_detail_logs = []
    try:
        search_detail_logs = conn.execute(
            """SELECT details, timestamp FROM agent_logs
               WHERE event_type = 'web_search_detail'
                 AND conversation_id = ?
               ORDER BY id""",
            (f"startup:{business_name}",),
        ).fetchall()
    except Exception:
        pass

    f.write(_line("─") + "\n")
    f.write(f"  🌐 WEB SEARCH KNOWLEDGE — {len(web_articles)} article(s) stored\n")
    f.write(_line("─") + "\n\n")

    if search_detail_logs:
        f.write(f"  Total searches done: {len(search_detail_logs)}\n\n")
        for idx, slog in enumerate(search_detail_logs, 1):
            try:
                det     = json.loads(slog["details"])
                query   = det.get("query", "(unknown)")
                results = det.get("results", [])
                ts      = (slog["timestamp"] or "")[:19]
                f.write(f"  Search #{idx}: \"{query}\"\n")
                f.write(f"      Results: {len(results)}   [{ts}]\n")
                for j, r in enumerate(results, 1):
                    title = (r.get("title") or "")[:70]
                    href  = r.get("href", "")
                    f.write(f"      Link {j}: {title}\n")
                    f.write(f"               {href}\n")
                f.write("\n")
            except Exception:
                pass
    elif web_articles:
        # No detail logs but articles exist — show topic summaries
        for i, art in enumerate(web_articles, 1):
            topic   = (art["topic"] or "").replace("[Web Knowledge] ", "")
            source  = art["source"] or "unknown"
            content = (art["content"] or "").strip()[:250].replace("\n", " ")
            src_lbl = "DDG live" if source == "web_search" else "LLM fallback"
            f.write(f"  [{i}] {topic}  ({src_lbl})\n")
            f.write(f"      {content}…\n\n")
    else:
        f.write("  (web search not yet run for this business)\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# One turn block
# ─────────────────────────────────────────────────────────────────────────────

def _write_turn(f, messages, tool_calls, agent_events, business_name, conversation_id):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    f.write("\n" + _line("═") + "\n")
    f.write(f"  TURN  |  {now}\n")
    f.write(f"  BUSINESS: {business_name or '(unknown)'}   CONV: {conversation_id[:20]}…\n")
    f.write(_line("═") + "\n\n")

    # ── Full conversation history ──────────────────────────────────────────
    f.write(_line("─") + "\n")
    f.write("  💬 CONVERSATION (full history this session)\n")
    f.write(_line("─") + "\n")
    for msg in messages:
        role    = (msg["role"] or "").upper()
        content = (msg["content"] or "").strip()
        ts      = msg["timestamp"] or ""
        icon    = "👤 USER" if role == "USER" else "🤖 AGENT"
        f.write(f"\n{icon}  [{ts}]\n{content}\n")
    f.write("\n")

    # ── Tool calls — full inputs & outputs ────────────────────────────────
    f.write(_line("─") + "\n")
    f.write("  🔧 TOOL CALLS THIS TURN\n")
    f.write(_line("─") + "\n")
    if not tool_calls:
        f.write("  (no tools called)\n\n")
    else:
        for i, tc in enumerate(tool_calls, 1):
            status = "✅ OK" if tc["activated"] else "⏭ SKIPPED"
            f.write(f"\n  Tool #{i}: {tc['tool_name']}  [{status}]  @ {tc['timestamp']}\n")
            f.write("  ▸ INPUTS:\n")
            for line in _fmt(tc["inputs"]).splitlines():
                f.write(f"      {line}\n")
            f.write("  ▸ OUTPUTS:\n")
            for line in _fmt(tc["outputs"]).splitlines():
                f.write(f"      {line}\n")
        f.write("\n")

    # ── Routing + LLM + DB events ─────────────────────────────────────────
    f.write(_line("─") + "\n")
    f.write("  🧠 AGENT REASONING  (routing · LLM calls · DB writes)\n")
    f.write(_line("─") + "\n")
    if not agent_events:
        f.write("  (no events)\n\n")
    else:
        for ev in agent_events:
            etype = ev["event_type"] or ""
            ts    = ev["timestamp"] or ""
            raw   = ev["details"] or "{}"

            if etype.startswith("routing"):     icon = "🗺"
            elif etype.startswith("llm_call"):  icon = "🤖"
            elif etype.startswith("db_write"):  icon = "💾"
            elif etype.startswith("reasoning"): icon = "💡"
            else:                               icon = "•"

            f.write(f"\n  {icon} [{ts}] {etype}\n")
            try:
                det = json.loads(raw)
                for k, v in (det.items() if isinstance(det, dict) else {}.items()):
                    val = str(v)
                    # Full prompt/response shown; other long values trimmed
                    if len(val) > 600 and k not in ("prompt", "response", "raw"):
                        val = val[:600] + "…"
                    f.write(f"      {k}: {val}\n")
            except Exception:
                f.write(f"      {str(raw)[:400]}\n")
    f.write("\n")


# ─────────────────────────────────────────────────────────────────────────────
# Public: append one turn  (called from agent.py node_log_turn)
# ─────────────────────────────────────────────────────────────────────────────

def append_turn_to_master_log(conversation_id: str, business_name: str = "") -> None:
    """
    Append the current turn to logs/all_logs.log.
    Writes a session header (with full web search summary + links) the first
    time a new conversation_id is seen in this process run.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    conn = _get_db()

    try:
        # Full conversation history so far (always show all)
        all_messages = conn.execute(
            """SELECT role, content, timestamp FROM conversations
               WHERE conversation_id = ? ORDER BY id""",
            (conversation_id,),
        ).fetchall()

        # ── Only tool calls from THIS turn ─────────────────────────────────
        # Use a FRESH load each time — never use a stale _cs that might
        # overwrite pending_booking or other keys set by tools.
        from core.database import load_conversation_state, save_conversation_state as _scs
        _log_cs  = load_conversation_state(conversation_id) or {}
        last_id  = int(_log_cs.get("_log_last_tool_id", 0))

        tool_calls = list(reversed(conn.execute(
            """SELECT id, tool_name, inputs, outputs, activated, timestamp
               FROM tool_calls
               WHERE conversation_id = ? AND id > ?
               ORDER BY id ASC LIMIT 10""",
            (conversation_id, last_id),
        ).fetchall()))

        # Persist new watermark — reload fresh again before saving
        if tool_calls:
            new_max  = max(int(tc["id"]) for tc in tool_calls)
            _log_cs2 = load_conversation_state(conversation_id) or {}
            _log_cs2["_log_last_tool_id"] = new_max
            _scs(conversation_id, _log_cs2)

        # ── Only reasoning events from THIS turn ────────────────────────────
        last_agent_id = int(_log_cs.get("_log_last_agent_id", 0))

        agent_events = list(reversed(conn.execute(
            """SELECT event_type, details, timestamp
               FROM agent_logs
               WHERE conversation_id = ? AND id > ?
                 AND event_type NOT LIKE 'llm_call:memory:%'
               ORDER BY id ASC LIMIT 30""",
            (conversation_id, last_agent_id),
        ).fetchall()))

        # Persist agent events watermark — reload fresh before saving
        try:
            max_agent_id = conn.execute(
                "SELECT MAX(id) FROM agent_logs WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0] or 0
            _log_cs3 = load_conversation_state(conversation_id) or {}
            _log_cs3["_log_last_agent_id"] = int(max_agent_id)
            _scs(conversation_id, _log_cs3)
        except Exception:
            pass

        with open(ALL_LOGS_FILE, "a", encoding="utf-8") as f:
            session_key = f"{conversation_id}:{business_name}"
            if session_key not in _session_headers_written:
                _write_session_header(f, business_name, conversation_id, conn)
                _session_headers_written.add(session_key)

            _write_turn(f, all_messages, tool_calls, agent_events,
                        business_name, conversation_id)

    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Public: store DDG search details so they appear in the session header
# ─────────────────────────────────────────────────────────────────────────────

def log_web_search_detail(business_name: str, query: str, results: list) -> None:
    """
    Store DDG query + result links in agent_logs.
    Called from knowledge_enricher after every successful DDG fetch.
    Shows up in the WEB SEARCH section of all_logs.log.
    """
    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO agent_logs (conversation_id, event_type, details, timestamp)
               VALUES (?, 'web_search_detail', ?, ?)""",
            (
                f"startup:{business_name}",
                json.dumps({
                    "business": business_name,
                    "query":    query,
                    "results":  [
                        {"title": r.get("title", ""), "href": r.get("href", "")}
                        for r in (results or [])
                    ],
                }, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception as exc:
        print(f"  [MASTER LOG] Could not store web search detail: {exc}")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone rebuild:  python -m core.master_log
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_master_log() -> str:
    os.makedirs(LOGS_DIR, exist_ok=True)
    conn = _get_db()
    try:
        conv_rows = conn.execute(
            """SELECT DISTINCT conversation_id, MIN(timestamp) as first_ts, business_name
               FROM conversations GROUP BY conversation_id ORDER BY first_ts"""
        ).fetchall()
    finally:
        conn.close()

    with open(ALL_LOGS_FILE, "w", encoding="utf-8") as f:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        f.write(_line("═") + "\n")
        f.write(f"  GENERIC AI AGENT — MASTER LOG\n")
        f.write(f"  Rebuilt: {now}   Conversations: {len(conv_rows)}\n")
        f.write(f"  File: {ALL_LOGS_FILE}\n")
        f.write(_line("═") + "\n")

    _session_headers_written.clear()
    for row in conv_rows:
        append_turn_to_master_log(row["conversation_id"], row["business_name"] or "")

    print(f"  [MASTER LOG] Rebuilt → {ALL_LOGS_FILE}  ({len(conv_rows)} conversations)")
    return ALL_LOGS_FILE


if __name__ == "__main__":
    import sys
    path = rebuild_master_log()
    print(f"Written: {path}")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatibility stubs (imported by older versions of main.py)
# ─────────────────────────────────────────────────────────────────────────────

def log_session_start(business_name: str = "", business_type: str = "", chunks: int = 0) -> None:
    """Log when a business session starts. Appended to all_logs.log."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(ALL_LOGS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{_line('-')}\n")
        f.write(f"  [SESSION START]  {now}\n")
        f.write(f"{_line('-')}\n\n")
        if business_name:
            f.write(f"  [{now}]  business_loaded  — {business_name} ({business_type})\n")
            f.write(f"      chunks: {chunks}\n\n")


def log_event(
    event_type: str = "",
    details: str = "",
    business_name: str = "",
    conversation_id: str = "",
    **kwargs,
) -> None:
    """Log a generic event to all_logs.log."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(ALL_LOGS_FILE, "a", encoding="utf-8") as f:
        label = f"  — {business_name}" if business_name else ""
        f.write(f"  [{now}]  {event_type}{label}\n")
        if details:
            for line in str(details).splitlines():
                f.write(f"      {line}\n")
        f.write("\n")