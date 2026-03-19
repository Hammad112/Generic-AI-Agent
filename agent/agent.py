"""
agent/agent.py
--------------
Main agent execution loop using LangGraph StateGraph.

Architecture:
  1. retrieve_history       -- load conversation history from DB
  2. retrieve_context       -- RAG: topics -> chunk retrieval -> composite prompt
  3. smart_route_intents    -- single LLM call -> list of 1-3 intents (multi-intent support)
  4. execute_tools          -- run each routed tool in sequence
  5. critic_validate        -- LLM critic checks response against PDF rules
  6. generate_response      -- llm_chat() produces the final natural-language response
  7. log_turn               -- persist user + assistant messages

Fixed bugs vs original:
  - Syntax error in f-string ({} -> empty dict example)
  - retrieve_relevant_chunks called with correct (topics, chunks) signature
  - log_chunk_retrieval called with correct (conv_id, query, count, ids) signature
  - state["prompt"] is now actually passed to llm_chat() for final response
  - LangGraph StateGraph is built and compiled (not just imported)
  - Multi-intent: one turn can route to 2 tools (e.g. remove + add, info + book)
  - apply_coupon removed from routing list (no implementation exists)
"""

import os
import uuid
import json
import re
import sqlite3
from typing import TypedDict, Annotated
import operator
from datetime import datetime

from langgraph.graph import StateGraph, END

from core.llm_client import llm_call
from core.database import (
    get_global_stats,
    get_business_meta,
    save_conversation_state,
    load_conversation_state,
    _get_db,
)
from core.logger import (
    log_message,
    log_tool_call,
    log_chunk_retrieval,
    log_agent_event,
    log_llm_call,
    log_agent_reasoning,
    get_conversation_history,
)
from agent.rag_engine import (
    chunk_query_into_topics,
    retrieve_relevant_chunks,
    get_customer_context,
    build_composite_prompt,
)
from agent.tools import TOOLS


# ─────────────────────────────────────────────────────────────────────────────
# Business mode helper
# ─────────────────────────────────────────────────────────────────────────────

_APPOINTMENT_KW = {
    "dental", "dentist", "clinic", "doctor", "medical", "health",
    "salon", "beauty", "spa", "massage", "therapy", "therapist",
    "physiotherapy", "physio", "chiropractor", "optometry", "optometrist",
    "veterinary", "vet", "lawyer", "legal", "accountant", "accounting",
    "gym", "fitness", "yoga", "pilates", "barber", "barbershop",
    "dermatology", "skin", "nail", "lash", "tattoo", "piercing",
    "photography", "photographer", "studio",
}
_ORDERING_KW = {
    "restaurant", "pizzeria", "pizza", "food", "cafe", "coffee",
    "bakery", "deli", "diner", "takeout", "takeaway", "fast food",
    "grocery", "catering", "delivery", "burger", "sushi",
}
_BOTH_KW = {
    "dry clean", "laundry", "cleaner", "alterations", "tailor",
    "florist", "flower", "car wash", "auto", "garage", "mechanic",
    "library",
}


def _get_business_mode(business_type: str) -> str:
    bt = (business_type or "").lower()
    if any(k in bt for k in _BOTH_KW):
        return "both"
    if any(k in bt for k in _APPOINTMENT_KW):
        return "appointment"
    if any(k in bt for k in _ORDERING_KW):
        return "ordering"
    return "both"


# ─────────────────────────────────────────────────────────────────────────────
# State schema
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    user_id: int
    conversation_id: str
    query: str
    business_name: str
    business_type: str
    all_chunks: list[dict]
    history: list[dict]
    current_cart: list[dict]
    retrieved_chunks: list[dict]
    composite_prompt: str
    routed_tools: list[str]      # list of 1-3 tool names
    tool_params_map: dict        # {tool_name: {param: value}}
    tool_results: dict           # {tool_name: result_dict}
    critic_ok: bool
    response: str


# ─────────────────────────────────────────────────────────────────────────────
# Cart loader
# ─────────────────────────────────────────────────────────────────────────────

def _get_cart_summary(user_id: int, conversation_id: str, business_name: str = "") -> list[dict]:
    conn = _get_db()
    try:
        if business_name:
            rows = conn.execute(
                """SELECT service_name AS name, quantity AS qty,
                          unit_price AS price, modifiers
                   FROM cart
                   WHERE user_id = ? AND session_id = ? AND business_name = ?
                   ORDER BY id""",
                (user_id, conversation_id, business_name),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT service_name AS name, quantity AS qty,
                          unit_price AS price, modifiers
                   FROM cart
                   WHERE user_id = ? AND session_id = ?
                   ORDER BY id""",
                (user_id, conversation_id),
            ).fetchall()
        items = [dict(r) for r in rows] if rows else []
        for i in items:
            i["price"] = float(i.get("price") or 0.0)
            i["qty"] = int(i.get("qty") or 1)
        return items
    except Exception:
        return []
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — retrieve history
# ─────────────────────────────────────────────────────────────────────────────

def node_retrieve_history(state: AgentState) -> AgentState:
    history = get_conversation_history(state["conversation_id"], limit=15)
    state["history"] = history
    state["current_cart"] = _get_cart_summary(
        state["user_id"], state["conversation_id"], state.get("business_name", "")
    )
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — retrieve context (RAG)
# ─────────────────────────────────────────────────────────────────────────────

def node_retrieve_context(state: AgentState) -> AgentState:
    query = state["query"]
    all_chunks = state["all_chunks"]
    conversation_id = state["conversation_id"]
    user_id = state["user_id"]
    business_name = state["business_name"]

    # LLM call I: topic extraction
    topics = chunk_query_into_topics(query, state["history"])
    log_agent_reasoning(conversation_id, "topic_extraction", {"query": query, "topics": topics})

    # LLM call III: chunk retrieval (topics + all_chunks from DB; enriched_knowledge from DB)
    retrieved, chunk_ids = retrieve_relevant_chunks(
        topics, all_chunks, top_n=5, business_name=business_name
    )

    # Customer context
    customer_context = get_customer_context(
        user_id,
        session_id=state["conversation_id"],
        business_name=business_name,
    )

    # Build composite prompt (used later in generate_response)
    prompt = build_composite_prompt(
        query=query,
        retrieved_chunks=retrieved,
        customer_context=customer_context,
        tool_results={},          # tools not run yet; filled in after
        conversation_history=state["history"],
        business_name=business_name,
        business_type=state["business_type"],
        current_cart=state["current_cart"],
    )

    # Log retrieval — correct 4-arg signature
    log_chunk_retrieval(conversation_id, query, len(retrieved), chunk_ids)

    state["retrieved_chunks"] = retrieved
    state["composite_prompt"] = prompt
    # Store customer_context in state for later nodes
    state["_customer_context"] = customer_context   # type: ignore[index]
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — smart multi-intent router
# ─────────────────────────────────────────────────────────────────────────────

_ORDERING_TOOLS = {
    "add_to_cart", "confirm_order", "remove_from_cart",
    "set_delivery_type", "set_delivery_address", "view_cart",
}
_APPOINTMENT_TOOLS = {
    "book_appointment", "reschedule_booking", "cancel_booking", "check_availability",
}


def _tool_allowed_for_mode(tool_name: str, business_mode: str) -> bool:
    if business_mode == "ordering" and tool_name in _APPOINTMENT_TOOLS:
        return False
    if business_mode == "appointment" and tool_name in _ORDERING_TOOLS:
        return False
    return True


def node_smart_route(state: AgentState) -> AgentState:
    """
    Route to 1-2 tools.
    Priority:
      1. Pending confirmation shortcuts (pending_booking / pending_cancel + yes)
      2. Greeting detection
      3. LLM-based intent routing
    """
    query         = state["query"]
    business_mode = _get_business_mode(state["business_type"])
    q_lower       = query.lower().strip()

    # ── Check conversation state for pending confirmations ──────────────────
    _cs = load_conversation_state(state["conversation_id"]) or {}
    pending_booking = _cs.get("pending_booking")
    pending_cancel  = _cs.get("pending_cancel")

    is_yes = any(w in q_lower for w in [
        "yes", "confirm", "go ahead", "sure", "ok", "okay", "correct",
        "please", "do it", "book it", "sounds good", "that's right",
        "cancel it", "yes cancel", "confirm cancel",
        "your recommended", "your suggestion", "go with that", "that works",
        "whatever you recommend",
    ])
    is_no = any(w in q_lower for w in ["no", "don't", "nope", "cancel", "stop", "nevermind", "never mind"])

    if pending_booking and is_yes:
        log_agent_event(state["conversation_id"], "routing_decision",
                        {"query": query, "routed_tools": ["book_appointment"],
                         "reason": "pending_booking+yes"})
        print("\n  [ROUTE] book_appointment  (pending_booking confirmed)")

        state["routed_tools"]    = ["book_appointment"]
        state["tool_params_map"] = {"book_appointment": {}}
        return state

    if pending_cancel and is_yes:
        log_agent_event(state["conversation_id"], "routing_decision",
                        {"query": query, "routed_tools": ["cancel_booking"],
                         "reason": "pending_cancel+yes"})
        print("\n  [ROUTE] cancel_booking  (pending_cancel confirmed)")

        state["routed_tools"]    = ["cancel_booking"]
        state["tool_params_map"] = {"cancel_booking": {}}
        return state

    if (pending_booking or pending_cancel) and is_no:
        # User said no to a pending confirmation — clear it and inform
        _cs.pop("pending_booking", None)
        _cs.pop("pending_cancel", None)
        save_conversation_state(state["conversation_id"], _cs)
        state["routed_tools"]    = ["get_business_info"]
        state["tool_params_map"] = {"get_business_info": {}}
        return state

    # ── Greeting shortcut (no LLM call needed) ───────────────────────────────
    GREETINGS = {"hello", "hi", "hey", "hiya", "howdy", "good morning",
                 "good afternoon", "good evening", "greetings", "sup", "yo"}
    if any(q_lower.startswith(g) or q_lower == g for g in GREETINGS):
        log_agent_event(state["conversation_id"], "routing_decision",
                        {"query": query, "routed_tools": ["greet_customer"],
                         "reason": "greeting_shortcut"})
        print("\n  [ROUTE] greet_customer  (greeting shortcut)")

        state["routed_tools"]    = ["greet_customer"]
        state["tool_params_map"] = {"greet_customer": {}}
        return state

    # ── "yes" after REPLACE/CORRECT offer: route to remove + add, not confirm_order ─
    last_agent = ""
    for m in reversed(state.get("history", [])[-6:]):
        if m.get("role") == "assistant":
            last_agent = (m.get("content") or "").lower()
            break
    replace_patterns = ("replace", "replace the", "replace x with", "swap", "change to", "correct that", "wrong item")
    if last_agent and any(p in last_agent for p in replace_patterns):
        _looks_yes = (
            any(re.search(rf"\b{re.escape(w)}\b", q_lower) for w in ["yes", "sure", "please", "replace", "do it"])
            or re.search(r"\b(ok|okay)\b", q_lower) is not None
        )
        if _looks_yes:
            log_agent_event(state["conversation_id"], "routing_decision",
                            {"query": query, "routed_tools": ["remove_from_cart", "add_to_cart"], "reason": "yes_after_replace_offer"})
            print("\n  [ROUTE] remove_from_cart + add_to_cart  (yes after replace offer)")
            state["routed_tools"] = ["remove_from_cart", "add_to_cart"]
            state["tool_params_map"] = {"remove_from_cart": {}, "add_to_cart": {}}
            return state

    # ── "yes" after ADD offer: route to add_to_cart, not confirm_order ──────
    add_offer_patterns = ("add ", "would you like to add", "add some ", "let's add", "for just $", "to complement", "to your order")
    if last_agent and any(p in last_agent for p in add_offer_patterns):
        # Use word boundaries to avoid "coke" matching "ok", "yesterday" matching "yes"
        _conf_phrases = ["yes", "add that", "add it", "sure", "please"]
        _conf_words = ["yes", "sure", "please"]
        _looks_confirm = (
            any(p in q_lower for p in ["add that", "add it"])
            or any(re.search(rf"\b{re.escape(w)}\b", q_lower) for w in _conf_words)
            or re.search(r"\b(ok|okay)\b", q_lower) is not None
        )
        if _looks_confirm:
            log_agent_event(state["conversation_id"], "routing_decision",
                            {"query": query, "routed_tools": ["add_to_cart"], "reason": "yes_after_add_offer"})
            print("\n  [ROUTE] add_to_cart  (yes after add offer)")
            state["routed_tools"] = ["add_to_cart"]
            state["tool_params_map"] = {"add_to_cart": {}}
            return state

    # ── LLM routing for everything else ─────────────────────────────────────
    recent_history = ""
    for msg in state["history"][-4:]:
        role    = msg.get("role", "").upper()
        content = msg.get("content", "")[:200]
        recent_history += f"{role}: {content}\n"

    cart_status = (f"{len(state['current_cart'])} items in cart"
                   if state["current_cart"] else "cart is empty")
    valid_tools = [t for t in TOOLS if _tool_allowed_for_mode(t, business_mode)]

    routing_prompt = (
        "You are an intent router for a customer service AI agent.\n\n"
        "CONTEXT:\n"
        f"- Business mode: {business_mode}\n"
        f"- Cart status: {cart_status}\n"
        f"- Recent conversation:\n{recent_history or 'No prior messages.'}\n\n"
        f'CUSTOMER MESSAGE: "{query}"\n\n'
        "AVAILABLE TOOLS (choose ONLY from this exact list):\n"
        + "\n".join(f"  - {t}" for t in valid_tools)
        + "\n\nROUTING RULES:\n"
        '  - "hello/hi/hey/good morning"                        -> [greet_customer]\n'
        '  - "change X to Y" / "replace X with Y"              -> [remove_from_cart, add_to_cart]\n'
        '  - "book at 3pm and how much does it cost"           -> [book_appointment, get_pricing]\n'
        '  - "yes/confirm" after an ORDER summary              -> [confirm_order]\n'
        '  - "yes/confirm" after a BOOKING proposal            -> [book_appointment]\n'
        '  - "yes/confirm" after a CANCEL prompt               -> [cancel_booking]\n'
        '  - customer wants to order/buy/add item              -> [add_to_cart]\n'
        '  - questions about hours/info/policies               -> [get_business_info]\n'
        '  - cancel appointment                                 -> [cancel_booking]\n'
        '  - reschedule / move appointment                     -> [reschedule_booking]\n'
        '  - question about price only                         -> [get_pricing]\n'
        '  - "cost breakdown" / "my order total" / "what did I order" -> [get_order_history]\n'
        '  - my bookings / past appointments / history         -> [get_order_history]\n\n'
        "RULES:\n"
        "  - tool MUST exactly match one from AVAILABLE TOOLS. No invented names.\n"
        "  - Do NOT use apply_coupon.\n"
        "  - confidence < 0.4 → use get_business_info.\n"
        "  - Return 1 intent normally, 2 only for genuinely dual-action messages.\n\n"
        "Return ONLY valid JSON. No markdown.\n"
        'Schema: {"intents": [{"tool": "greet_customer", "confidence": 0.97,'
        ' "params": {}, "reasoning": "customer said hello"}]}'
    )

    raw = llm_call(routing_prompt, temperature=0.0, max_tokens=350)
    log_agent_reasoning(state["conversation_id"], "routing", {"query": query, "raw": raw[:300]})

    routed_tools: list[str] = []
    tool_params_map: dict = {}

    try:
        data = json.loads(raw)
        intents = data.get("intents", [])
        for intent in intents[:2]:          # max 2 tools per turn
            tool_name = intent.get("tool", "")
            confidence = float(intent.get("confidence", 0.0))
            params = intent.get("params", {})

            if tool_name not in TOOLS:
                tool_name = "get_business_info"
            if not _tool_allowed_for_mode(tool_name, business_mode):
                tool_name = "get_business_info"
            if confidence < 0.35:
                tool_name = "get_business_info"

            routed_tools.append(tool_name)
            tool_params_map[tool_name] = params if isinstance(params, dict) else {}
    except (json.JSONDecodeError, ValueError, TypeError):
        routed_tools = ["get_business_info"]
        tool_params_map = {"get_business_info": {}}

    if not routed_tools:
        routed_tools = ["get_business_info"]
        tool_params_map = {"get_business_info": {}}

    log_agent_event(state["conversation_id"], "routing_decision", {
        "query": query,
        "routed_tools": routed_tools,
        "business_mode": business_mode,
    })

    # Live console — show routing decision
    tools_display = " + ".join(routed_tools)
    print("\n  [ROUTE] " + tools_display + "  (mode=" + business_mode + ")")

    state["routed_tools"]    = routed_tools
    state["tool_params_map"] = tool_params_map
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — execute tools
# ─────────────────────────────────────────────────────────────────────────────

def node_execute_tools(state: AgentState) -> AgentState:
    """Execute each routed tool in sequence and collect results."""
    tool_results: dict = {}

    for tool_name in state["routed_tools"]:
        extra_params   = state["tool_params_map"].get(tool_name, {})
        state_for_tool = {**state, **extra_params}

        # ── Live console log before execution ───────────────────────────
        params_display = (
            ", ".join(k + "=" + repr(v)[:40] for k, v in extra_params.items())
            if extra_params else "(none)"
        )
        print("\n  [TOOL] " + tool_name + "  params=" + params_display)

        try:
            if tool_name in TOOLS:
                tool_func, _ = TOOLS[tool_name]
                result = tool_func(state_for_tool)
            else:
                result = {"success": False, "message": "Unknown tool: " + tool_name}
        except Exception as exc:
            result = {
                "success": False,
                "error":   str(exc),
                "message": "An error occurred processing your request. Please try again.",
            }
            print("  [TOOL ERROR] " + tool_name + ": " + str(exc))

        # ── Live console log after execution ────────────────────────────
        success     = result.get("success", False)
        status_icon = "OK" if success else "FAIL"
        short_msg   = result.get("message", "")
        display_msg = (short_msg[:120] + "...") if len(short_msg) > 120 else short_msg
        extra_tags  = ""
        if result.get("needs_confirmation") or result.get("needs_info"):
            extra_tags += " [NEEDS CONFIRMATION]"
        if result.get("conflict"):
            extra_tags += " [CONFLICT]"
        print("  " + status_icon + " success=" + str(success) + extra_tags)
        for msg_line in display_msg.split("\n"):
            if msg_line.strip():
                print("     " + msg_line)

        log_tool_call(state["conversation_id"], tool_name, extra_params, result)
        tool_results[tool_name] = result

        # Refresh current_cart from tool result if it returned cart data
        if result.get("cart_items") is not None and "cart_total" in result:
            state["current_cart"] = [
                {"name": c.get("name", ""), "qty": int(c.get("qty", 1)), "price": float(c.get("price", 0.0)), "modifiers": c.get("modifiers", "")}
                for c in result["cart_items"]
            ]

    state["tool_results"] = tool_results
    return state
def node_critic(state: AgentState) -> AgentState:
    """
    LLM Critic: validates that tool results are consistent with PDF business rules.
    If a rule violation is detected the response generation will be warned.
    """
    # Quick pass: if all tools succeeded, critic is not needed for speed
    all_success = all(
        r.get("success", False) for r in state["tool_results"].values()
    )
    if all_success:
        state["critic_ok"] = True
        return state

    # Check failed tools for policy violations
    business_name = state["business_name"]
    hours_meta = get_business_meta("business_hours") or ""

    failed_messages = [
        f"{tool}: {result.get('message', '')}"
        for tool, result in state["tool_results"].items()
        if not result.get("success")
    ]

    critic_prompt = f"""You are a quality-control critic for a customer service AI agent for "{business_name}".

Business hours/policies known:
{hours_meta[:500] or "Not specified."}

The following tool calls FAILED this turn:
{chr(10).join(failed_messages)}

Customer query: "{state['query']}"

Is the failure due to:
A) A genuine rule/policy violation (e.g. closed on Sunday, booking too far out)?
B) Missing information the customer must provide?
C) A technical error that should be retried?

Return ONLY JSON:
{{"violation_type": "A|B|C", "explanation": "brief reason", "ok_to_proceed": true}}

No markdown. Just JSON."""

    raw = llm_call(critic_prompt, temperature=0.0, max_tokens=150)
    log_agent_reasoning(state["conversation_id"], "critic", {"raw": raw[:200]})

    try:
        data = json.loads(raw)
        state["critic_ok"] = data.get("ok_to_proceed", True)
    except Exception:
        state["critic_ok"] = True   # On parse failure, proceed

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — generate final response
# ─────────────────────────────────────────────────────────────────────────────

def node_generate_response(state: AgentState) -> AgentState:
    """
    THE CRITICAL MISSING STEP in the original code.
    Pass the composite prompt + tool results to llm_chat() to get a
    natural, conversational, grounded response.
    """
    customer_context = state.get("_customer_context", {})   # type: ignore[attr-defined]

    # Rebuild prompt now that we have tool results
    final_prompt = build_composite_prompt(
        query=state["query"],
        retrieved_chunks=state["retrieved_chunks"],
        customer_context=customer_context,
        tool_results=state["tool_results"],
        conversation_history=state["history"],
        business_name=state["business_name"],
        business_type=state["business_type"],
        current_cart=state["current_cart"],
    )

    # Add critic warning if rules were violated
    if not state.get("critic_ok", True):
        final_prompt += (
            "\n\nCRITIC WARNING: A tool failed due to a policy or rule violation. "
            "Explain this clearly to the customer. Do not bypass the policy."
        )

    response_text = llm_call(final_prompt, temperature=0.7, max_tokens=700)

    # Log the LLM call for traceability
    log_llm_call(
        state["conversation_id"],
        call_type="generate_response",
        prompt_snippet=final_prompt[:500],
        response_snippet=response_text[:500],
        model=os.getenv("GEMINI_MODEL", os.getenv("OPENAI_MODEL", "unknown")),
    )

    state["response"] = response_text
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — log turn
# ─────────────────────────────────────────────────────────────────────────────

def node_log_turn(state: AgentState) -> AgentState:
    business_name = state["business_name"]
    log_message(state["conversation_id"], "user", state["query"], business_name)
    log_message(state["conversation_id"], "assistant", state["response"], business_name)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Build the LangGraph
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_history",  node_retrieve_history)
    graph.add_node("retrieve_context",  node_retrieve_context)
    graph.add_node("smart_route",       node_smart_route)
    graph.add_node("execute_tools",     node_execute_tools)
    graph.add_node("critic",            node_critic)
    graph.add_node("generate_response", node_generate_response)
    graph.add_node("log_turn",          node_log_turn)

    graph.set_entry_point("retrieve_history")
    graph.add_edge("retrieve_history",  "retrieve_context")
    graph.add_edge("retrieve_context",  "smart_route")
    graph.add_edge("smart_route",       "execute_tools")
    graph.add_edge("execute_tools",     "critic")
    graph.add_edge("critic",            "generate_response")
    graph.add_edge("generate_response", "log_turn")
    graph.add_edge("log_turn",          END)

    return graph.compile()


_AGENT_GRAPH = _build_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_turn(
    user_id: int,
    conversation_id: str,
    query: str,
    all_chunks: list[dict],
    business_name: str,
    business_type: str,
) -> dict:
    """
    Execute one conversational turn through the LangGraph agent.
    Called from main.py and app/server.py.
    Returns: {"response": str, "tool_results": dict, "ics_file_url": str}

    Memory: Conversation history is loaded from DB (conversations table) each turn.
    Context: PDF chunks are reloaded from DB (pdf_chunks table) each turn for freshness.
    """
    # Reload chunks from DB each turn so context is always fresh (not stale in-memory cache)
    from processing.pdf_processor import load_chunks
    try:
        all_chunks = load_chunks(business_name) or all_chunks
    except Exception:
        pass  # fallback to passed-in chunks if DB load fails

    initial_state: AgentState = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "query": query,
        "business_name": business_name,
        "business_type": business_type,
        "all_chunks": all_chunks,
        "history": [],
        "current_cart": [],
        "retrieved_chunks": [],
        "composite_prompt": "",
        "routed_tools": [],
        "tool_params_map": {},
        "tool_results": {},
        "critic_ok": True,
        "response": "",
    }

    try:
        final_state = _AGENT_GRAPH.invoke(initial_state)
    except Exception as exc:
        print(f"[AGENT] Graph execution error: {exc}")
        import traceback
        traceback.print_exc()
        return {
            "response": "I'm sorry, something went wrong. Please try again.",
            "tool_results": {},
            "ics_file_url": "",
        }

    # Extract ICS url if any tool produced one
    ics_url = ""
    for result in final_state.get("tool_results", {}).values():
        if isinstance(result, dict) and result.get("ics_file_url"):
            ics_url = result["ics_file_url"]
            break

    return {
        "response": final_state.get("response", ""),
        "tool_results": final_state.get("tool_results", {}),
        "ics_file_url": ics_url,
    }