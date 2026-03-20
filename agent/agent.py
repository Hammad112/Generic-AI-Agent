"""
agent/agent.py
--------------
AI Agent using native LLM function calling.

Architecture (single LangGraph node replaces old route→execute→generate):

  1. retrieve_history    — load DB history + long-term user memory
  2. retrieve_context    — RAG: get relevant knowledge chunks
  3. function_call_loop  — single LLM call with ALL tools defined
                           LLM decides which tool(s) to call (or none)
                           Tools are executed; result fed back to LLM
                           LLM generates final natural response
  4. extract_memory      — extract new facts from conversation → user_memory
  5. log_turn            — persist messages + write log files

Benefits vs old routing approach:
  - LLM sees full history as proper chat messages (not squashed text)
  - LLM decides tools from context — no hardcoded shortcuts to break
  - Memory persists across sessions via user_memory table
  - Fewer LLM calls (routing + generation merged into one interaction)
"""

import os
import json
import re
import uuid
from typing import TypedDict, Annotated
import operator
from datetime import datetime

from langgraph.graph import StateGraph, END

from core.llm_client import llm_call, llm_with_tools
from core.database import (
    get_global_stats,
    get_business_meta,
    save_conversation_state,
    load_conversation_state,
    load_user_memory,
    save_user_memory_facts,
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
from agent.tool_schemas import OPENAI_TOOLS


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    user_id:          int
    conversation_id:  str
    query:            str
    business_name:    str
    business_type:    str
    all_chunks:       list[dict]
    history:          list[dict]       # short-term: last 15 turns from DB
    user_memory:      list[dict]       # long-term: persistent facts
    current_cart:     list[dict]
    retrieved_chunks: list[dict]
    knowledge_text:   str              # formatted knowledge for system prompt
    customer_context: dict
    tool_results:     dict
    critic_ok:        bool
    response:         str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_APPOINTMENT_KW = {
    "dental","dentist","clinic","doctor","medical","health",
    "salon","beauty","spa","massage","therapy","therapist",
    "physiotherapy","physio","chiropractor","optometry","optometrist",
    "veterinary","vet","lawyer","legal","accountant","accounting",
    "gym","fitness","yoga","pilates","barber","barbershop",
    "dermatology","skin","nail","lash","tattoo","piercing",
    "photography","photographer","studio",
}
_ORDERING_KW = {
    "restaurant","pizzeria","pizza","food","cafe","coffee",
    "bakery","deli","diner","takeout","takeaway","fast food",
    "grocery","catering","delivery","burger","sushi",
}
_BOTH_KW = {
    "dry clean","laundry","cleaner","alterations","tailor",
    "florist","flower","car wash","auto","garage","mechanic",
}

def _get_business_mode(business_type: str) -> str:
    bt = (business_type or "").lower()
    if any(k in bt for k in _BOTH_KW):      return "both"
    if any(k in bt for k in _APPOINTMENT_KW): return "appointment"
    if any(k in bt for k in _ORDERING_KW):    return "ordering"
    return "both"


def _get_cart_summary(user_id: int, conversation_id: str, business_name: str = "") -> list[dict]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT service_name AS name, quantity AS qty,
                      unit_price AS price, modifiers
               FROM cart
               WHERE user_id = ? AND session_id = ? AND business_name = ?
               ORDER BY id""",
            (user_id, conversation_id, business_name),
        ).fetchall()
        items = [dict(r) for r in rows] if rows else []
        for i in items:
            i["price"] = float(i.get("price") or 0.0)
            i["qty"]   = int(i.get("qty") or 1)
        return items
    except Exception:
        return []
    finally:
        conn.close()


def _build_system_prompt(state: AgentState) -> str:
    """
    Build the rich system prompt that the LLM sees at the start of every turn.
    Includes: business context, knowledge, customer profile, long-term memory, cart.
    """
    today_str       = datetime.now().strftime("%Y-%m-%d (%A)")
    business_name   = state["business_name"]
    business_type   = state["business_type"]
    business_mode   = _get_business_mode(business_type)

    # ── Knowledge base ───────────────────────────────────────────────────────
    knowledge = state.get("knowledge_text") or "No specific knowledge retrieved."

    # ── Customer profile ─────────────────────────────────────────────────────
    ctx = state.get("customer_context") or {}
    name_str  = ctx.get("full_name", "")
    phone_str = ctx.get("phone", "")
    loyalty   = ctx.get("loyalty", {})
    pts       = loyalty.get("points", 0) if loyalty else 0
    tier      = loyalty.get("tier", "bronze") if loyalty else "bronze"

    profile_lines = []
    if name_str:
        profile_lines.append(f"Name: {name_str}")
    if phone_str:
        profile_lines.append(f"Phone: {phone_str}")
    if pts:
        profile_lines.append(f"Loyalty: {pts} pts ({tier})")
    fav = ctx.get("usual_favorite", "")
    if fav:
        profile_lines.append(f"Favourite: {fav}")
    complaints = ctx.get("recent_complaints", [])
    if complaints:
        profile_lines.append(f"Past complaints: {len(complaints)}")
    profile_text = "\n".join(profile_lines) if profile_lines else "Guest customer."

    # ── Long-term memory ─────────────────────────────────────────────────────
    mem_facts = state.get("user_memory") or []
    if mem_facts:
        mem_lines = [f"  [{f['category']}] {f['fact']}" for f in mem_facts[:15]]
        memory_text = "WHAT WE KNOW ABOUT THIS CUSTOMER (from past sessions):\n" + "\n".join(mem_lines)
    else:
        memory_text = ""

    # ── Cart ─────────────────────────────────────────────────────────────────
    cart = state.get("current_cart") or []
    if cart:
        total = round(sum(int(i.get("qty", 1)) * float(i.get("price", 0)) for i in cart), 2)
        cart_lines = "\n".join(
            f"  - {int(i.get('qty',1))}x {i.get('name','?')} @ ${float(i.get('price',0)):.2f}"
            + (f" ({i['modifiers']})" if i.get("modifiers") else "")
            for i in cart
        )
        cart_text = f"CURRENT CART:\n{cart_lines}\n  Subtotal: ${total:.2f}"
    else:
        cart_text = "CART: Empty"

    # ── Mode restrictions ────────────────────────────────────────────────────
    if business_mode == "appointment":
        mode_note = "This is an appointment-only business. Do NOT offer add-to-cart or delivery."
    elif business_mode == "ordering":
        mode_note = "This is an ordering/food business. Do NOT offer appointment booking."
    else:
        mode_note = "This business supports both ordering and appointments."

    system = f"""You are the AI customer service agent for "{business_name}" ({business_type}).
Today: {today_str}
{mode_note}

=== KNOWLEDGE BASE ===
{knowledge}

=== CUSTOMER PROFILE ===
{profile_text}

{memory_text}

=== {cart_text} ===

CRITICAL RESPONSE RULES:
1. GROUNDING: Only state facts from the KNOWLEDGE BASE or TOOL RESULTS. Never invent prices, dates, names, or policies.
2. TOOL SUCCESS: When a tool returns success=true, the action IS done. State the outcome. Never re-ask.
3. TOOL FAILURE: Explain the specific failure reason warmly. Do NOT bypass it.
4. CONFIRMATION: If tool returns needs_confirmation=true, present the summary and ask the customer to confirm. Tell them: "Type **'yes'** to confirm or **'no'** to cancel."
5. CART: The CURRENT CART above is authoritative. Never add items from memory. If cart is empty, say empty.
6. NAMES: Use the customer's real name naturally. Never use [Name] or [Customer].
7. HISTORY: Reference what the customer said earlier in this conversation when relevant.
8. UPSELL: Suggest ONE related item per session naturally.
9. CONCISE: Friendly but concise. Skip filler phrases like "Certainly!" or "Of course!".
10. MEMORY: Use the WHAT WE KNOW section to personalise responses even at the start of a new session.
11. APPOINTMENTS: For booking, ALWAYS confirm: service, date, time, provider (if specified). Never book without all three.
12. PRICING CONTEXT: When customer narrows to a category (e.g. "feet services"), remember that context for follow-up questions like "cheapest" or "tell me more".
13. EXACT WORDS — CRITICAL: Whenever you need the customer to choose or confirm, ALWAYS tell them the EXACT WORD to type. Never ask a vague question when specific options exist.
    - Delivery choice: "Type **pickup** to collect, or **delivery** to have it sent to you."
    - Confirmation: "Type **yes** to confirm or **no** to cancel."
    - Date/time: "Reply with a date and time, e.g. **Monday at 2pm** or **March 25 at 10am**."
    - Service choice: "Type the service name, e.g. **Dental Cleaning** or **Root Canal**."
    - Cancel which booking: "Type the booking reference, e.g. **APPT-00087**, or say **latest** for your most recent one."
14. TOOL NEEDS_INFO: If a tool returns success=false and needs_info=true, restate it as clear instructions with the exact input format the customer should use. Do NOT repeat the tool's raw message — rephrase it with bold **exact words**.
"""
    return system


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — retrieve history + memory
# ─────────────────────────────────────────────────────────────────────────────

def node_retrieve_history(state: AgentState) -> AgentState:
    # Short-term: last 15 messages this session
    state["history"] = get_conversation_history(state["conversation_id"], limit=15)
    # Cart
    state["current_cart"] = _get_cart_summary(
        state["user_id"], state["conversation_id"], state.get("business_name", "")
    )
    # Long-term memory: facts about this user from all past sessions
    state["user_memory"] = load_user_memory(state["user_id"], state["business_name"])
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — retrieve context (RAG)
# ─────────────────────────────────────────────────────────────────────────────

def node_retrieve_context(state: AgentState) -> AgentState:
    query        = state["query"]
    all_chunks   = state["all_chunks"]
    conv_id      = state["conversation_id"]
    business_name = state["business_name"]

    topics = chunk_query_into_topics(query, state["history"])
    log_agent_reasoning(conv_id, "topic_extraction", {"query": query, "topics": topics})

    retrieved, chunk_ids = retrieve_relevant_chunks(topics, all_chunks, top_n=6, business_name=business_name)
    log_chunk_retrieval(conv_id, query, len(retrieved), chunk_ids)

    customer_context = get_customer_context(
        state["user_id"], session_id=conv_id, business_name=business_name
    )

    # Format knowledge text for system prompt
    if retrieved:
        knowledge_text = "\n\n".join(
            f"[{c.get('section_title','Info')}]\n{c.get('text','')[:500]}"
            for c in retrieved
        )
    else:
        knowledge_text = "No specific knowledge retrieved for this query."

    state["retrieved_chunks"] = retrieved
    state["knowledge_text"]   = knowledge_text
    state["customer_context"] = customer_context
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — function call loop (replaces route + execute + generate)
# ─────────────────────────────────────────────────────────────────────────────

def _exact_word_guidance(tool_name: str, result: dict) -> str:
    """
    Return a clear instruction with the EXACT WORDS the customer should type,
    based on what tool failed and why.
    Called when the same tool fails 2+ times in a row.
    """
    guides = {
        "set_delivery_type": (
            "Please type exactly **pickup** (to collect your order yourself) "
            "or **delivery** (to have it sent to your address)."
        ),
        "confirm_order": (
            "Please type **confirm** to place your order, or **cancel** to discard it."
        ),
        "book_appointment": (
            "To book, please reply with the service name, date, and time. "
            "For example: **Dental Cleaning on Monday at 10am**"
        ),
        "reschedule_booking": (
            "Please reply with your new preferred date and time. "
            "For example: **Wednesday at 3pm** or **March 25 at 2pm**"
        ),
        "cancel_booking": (
            "Please type your booking reference (e.g. **APPT-00087**), "
            "or type **latest** to cancel your most recent appointment."
        ),
        "set_delivery_address": (
            "Please provide your full address including street, city, and postal code. "
            "For example: **123 Main Street, Toronto, ON M5V 1A1**"
        ),
        "validate_address": (
            "Please provide your full address. "
            "Example: **123 Main Street, Toronto, ON M5V 1A1**"
        ),
        "add_to_cart": (
            "Please type the name of the item you want to add, "
            "for example: **Margherita Large** or **Pepperoni Medium**"
        ),
        "remove_from_cart": (
            "Please type the exact name of the item to remove from your cart."
        ),
        "apply_loyalty_discount": (
            "To redeem your points, type **redeem points**."
        ),
        "handle_dispute": (
            "Please describe your issue in a sentence, e.g. **my pizza was cold** or **wrong item delivered**."
        ),
    }
    # Fall back to generic guidance using the tool's own message
    specific = guides.get(tool_name, "")
    if specific:
        return specific
    msg = result.get("message", "")
    if msg:
        return f"Please respond with the specific information needed. {msg}"
    return ""


def node_function_call_loop(state: AgentState) -> AgentState:
    """
    Single node that:
    1. Builds messages from history + system prompt
    2. Calls LLM with all tools defined — LLM decides which to call
    3. If tool called: executes it, adds result, calls LLM again for response
    4. Returns final natural language response

    The LLM now has full conversation context as proper messages,
    not squashed text — so it naturally remembers prior context.
    """
    conv_id       = state["conversation_id"]
    query         = state["query"]
    business_mode = _get_business_mode(state["business_type"])

    # ── Build system prompt ──────────────────────────────────────────────────
    system_prompt = _build_system_prompt(state)

    # ── Build message history for LLM ───────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history as proper chat messages
    for msg in state.get("history", []):
        role    = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if role == "assistant":
            role = "assistant"
        elif role == "user":
            role = "user"
        else:
            continue
        if content:
            messages.append({"role": role, "content": content})

    # Add the current user message
    messages.append({"role": "user", "content": query})

    # ── Filter tools by business mode ────────────────────────────────────────
    _ORDERING_TOOLS = {
        "add_to_cart","confirm_order","remove_from_cart",
        "set_delivery_type","set_delivery_address","view_cart",
    }
    _APPOINTMENT_TOOLS = {
        "book_appointment","reschedule_booking","cancel_booking","check_availability",
    }

    def _allowed(tool_name: str) -> bool:
        if business_mode == "ordering" and tool_name in _APPOINTMENT_TOOLS:
            return False
        if business_mode == "appointment" and tool_name in _ORDERING_TOOLS:
            return False
        return True

    filtered_tools = [
        t for t in OPENAI_TOOLS if _allowed(t["function"]["name"])
    ]

    # ── First LLM call: decide tools ─────────────────────────────────────────
    print(f"\n  [FC] Calling LLM with {len(filtered_tools)} tools...")
    response_text, tool_calls = llm_with_tools(
        messages=messages,
        tools=filtered_tools,
        conversation_id=conv_id,
        temperature=0.3,
        max_tokens=800,
    )

    tool_results = {}

    # ── Track repeated tool failures so we can break the loop ───────────────
    _cs       = load_conversation_state(conv_id) or {}
    fail_cnt  = _cs.get("_tool_fail_counts", {})   # {tool_name: consecutive_fail_count}

    if tool_calls:
        # ── Execute each tool ────────────────────────────────────────────────
        tool_results_msgs = []  # for second LLM call

        for tc in tool_calls[:3]:  # max 3 tools per turn
            tool_name = tc.get("name", "")
            tool_args = tc.get("arguments", {})
            tool_id   = tc.get("id", f"call_{tool_name}")

            print(f"\n  [TOOL] {tool_name}  args={list(tool_args.keys())}")

            if tool_name not in TOOLS:
                result = {"success": False, "message": f"Unknown tool: {tool_name}"}
            else:
                tool_func, _ = TOOLS[tool_name]
                # Merge tool args into state for the tool to use
                state_for_tool = {**state, **tool_args}
                try:
                    result = tool_func(state_for_tool)
                except Exception as exc:
                    result = {
                        "success": False,
                        "error": str(exc),
                        "message": "An error occurred. Please try again.",
                    }
                    print(f"  [TOOL ERROR] {tool_name}: {exc}")

            # Log tool call
            from core.tool_log_writer import print_tool_log, append_tool_log as _atl
            print_tool_log(tool_name, tool_args, result, activated=True)
            log_tool_call(conv_id, tool_name, tool_args, result)
            _atl(
                tool_name=tool_name,
                conversation_id=conv_id,
                timestamp=datetime.now().isoformat(),
                inputs=tool_args,
                outputs=result,
                activated=True,
            )

            log_agent_event(conv_id, "tool_executed", {
                "tool": tool_name, "success": result.get("success"), "args": tool_args
            })

            # ── Failure loop tracker ──────────────────────────────────────────
            if not result.get("success") and (result.get("needs_info") or result.get("needs_confirmation")):
                fail_cnt[tool_name] = fail_cnt.get(tool_name, 0) + 1
            else:
                fail_cnt[tool_name] = 0   # reset on success

            # If same tool failed 2+ times in a row, inject exact-word guidance
            if fail_cnt.get(tool_name, 0) >= 2:
                guidance = _exact_word_guidance(tool_name, result)
                if guidance:
                    result = {**result, "exact_word_guidance": guidance}
                    print(f"  [LOOP BREAK] {tool_name} failed {fail_cnt[tool_name]}x — injecting guidance")

            # ── CRITICAL: reload conversation_state AFTER tool ran ─────────────
            # The tool (e.g. book_appointment) may have written pending_booking to
            # conversation_state. If we use our stale _cs copy and save it back,
            # we overwrite whatever the tool just saved (wiping pending_booking).
            # Solution: always reload fresh from DB, then merge in fail_counts only.
            _fresh_cs = load_conversation_state(conv_id) or {}
            _fresh_cs["_tool_fail_counts"] = fail_cnt
            save_conversation_state(conv_id, _fresh_cs)

            tool_results[tool_name] = result

            # Refresh cart if tool returned cart data
            if result.get("cart_items") is not None and "cart_total" in result:
                state["current_cart"] = [
                    {"name": c.get("name",""), "qty": int(c.get("qty",1)),
                     "price": float(c.get("price",0.0)), "modifiers": c.get("modifiers","")}
                    for c in result["cart_items"]
                ]

            # Format tool result as a message for the second LLM call
            result_text = json.dumps(result, default=str, ensure_ascii=False)
            tool_results_msgs.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": result_text,
            })

        # ── Second LLM call: generate natural response using tool results ────
        # Rebuild messages with the assistant's tool call + tool results appended
        messages_with_results = messages + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.get("id", f"call_{tc.get('name','')}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                    }
                    for tc in tool_calls[:3]
                ],
            }
        ] + tool_results_msgs

        print(f"\n  [FC] Generating response with tool results...")
        final_text, _ = llm_with_tools(
            messages=messages_with_results,
            tools=[],           # no tools on second call — just generate text
            conversation_id=conv_id,
            temperature=0.7,
            max_tokens=700,
        )

        # If function calling second pass returned empty (Gemini sometimes),
        # fall back to build_composite_prompt path
        if not final_text.strip():
            final_text = _fallback_response(state, tool_results, system_prompt)

        response_text = final_text

    elif not response_text.strip():
        # LLM returned nothing — fallback
        response_text = _fallback_response(state, {}, system_prompt)

    # Log routing decision
    tool_names_used = [tc.get("name") for tc in tool_calls] if tool_calls else ["(no tool)"]
    log_agent_event(conv_id, "routing_decision", {
        "query": query,
        "routed_tools": tool_names_used,
        "business_mode": business_mode,
    })
    print(f"\n  [ROUTE] {', '.join(tool_names_used)}  (mode={business_mode})")

    state["tool_results"] = tool_results
    state["response"]     = response_text
    return state


def _fallback_response(state: AgentState, tool_results: dict, system_prompt: str) -> str:
    """
    Generate a response using the old prompt-based approach as a fallback
    when function calling returns empty or fails.
    """
    from agent.rag_engine import build_composite_prompt
    prompt = build_composite_prompt(
        query=state["query"],
        retrieved_chunks=state.get("retrieved_chunks", []),
        customer_context=state.get("customer_context", {}),
        tool_results=tool_results,
        conversation_history=state.get("history", []),
        business_name=state["business_name"],
        business_type=state["business_type"],
        current_cart=state.get("current_cart", []),
    )
    return llm_call(prompt, temperature=0.7, max_tokens=700,
                    conversation_id=state["conversation_id"])


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — extract memory facts from this turn
# ─────────────────────────────────────────────────────────────────────────────

def node_extract_memory(state: AgentState) -> AgentState:
    """
    After every turn, ask a small LLM call to extract any NEW facts worth
    remembering about this customer permanently.
    Only runs if there's a meaningful user message.
    """
    query    = state.get("query", "").strip()
    response = state.get("response", "").strip()
    history  = state.get("history", [])

    if len(query) < 10:   # Skip trivial messages
        return state

    # Build a short context for extraction
    recent = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}"
        for m in history[-4:]
    )

    extract_prompt = f"""You are extracting LONG-TERM memory facts from a customer service conversation.

Business: {state['business_name']} ({state['business_type']})

Recent conversation:
{recent}
USER: {query}
AGENT: {response[:300]}

Extract 0-3 important facts worth remembering about THIS CUSTOMER across future sessions.

Good facts to remember:
- Health conditions relevant to services ("has diabetes", "allergic to latex")
- Strong preferences ("prefers morning appointments", "likes thin crust")
- Family info ("has a daughter who needs cleaning too")
- Service interests ("interested in teeth whitening")
- Complaints / bad experiences ("had a bad experience with X")
- Lifestyle ("vegetarian", "student budget")

Do NOT extract:
- Generic facts about the business
- Transient info (today's specific booking details)
- Things already in current session context

Return ONLY a JSON array (empty [] if nothing worth saving):
[{{"category": "preference|health|family|service|complaint|personal", "fact": "short fact string"}}]
No markdown, just JSON."""

    try:
        raw = llm_call(extract_prompt, temperature=0.1, max_tokens=200,
                       conversation_id=f"memory:{state['conversation_id']}")
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        facts = json.loads(raw)
        if isinstance(facts, list) and facts:
            save_user_memory_facts(state["user_id"], state["business_name"], facts)
            print(f"\n  🧠 [MEMORY] Saved {len(facts)} new fact(s) for user {state['user_id']}")
    except Exception:
        pass  # Memory extraction is non-critical — never break the turn

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — log turn
# ─────────────────────────────────────────────────────────────────────────────

def node_log_turn(state: AgentState) -> AgentState:
    business_name = state["business_name"]
    log_message(state["conversation_id"], "user",      state["query"],    business_name)
    log_message(state["conversation_id"], "assistant", state["response"], business_name)

    try:
        from core.tool_log_writer import write_conversation_log
        log_path = write_conversation_log(
            conversation_id=state["conversation_id"],
            business_name=business_name,
            label=business_name,
        )
        print(f"\n  📄 [LOG] {log_path}")
    except Exception as exc:
        print(f"\n  [LOG WARN] {exc}")

    try:
        from core.master_log import append_turn_to_master_log
        append_turn_to_master_log(state["conversation_id"], business_name)
    except Exception as exc:
        print(f"\n  [LOG WARN] master log: {exc}")

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_history",     node_retrieve_history)
    graph.add_node("retrieve_context",     node_retrieve_context)
    graph.add_node("function_call_loop",   node_function_call_loop)
    graph.add_node("extract_memory",       node_extract_memory)
    graph.add_node("log_turn",             node_log_turn)

    graph.set_entry_point("retrieve_history")
    graph.add_edge("retrieve_history",   "retrieve_context")
    graph.add_edge("retrieve_context",   "function_call_loop")
    graph.add_edge("function_call_loop", "extract_memory")
    graph.add_edge("extract_memory",     "log_turn")
    graph.add_edge("log_turn",           END)

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
    Execute one conversational turn. Called from main.py and app/server.py.
    Returns: {"response": str, "tool_results": dict, "ics_file_url": str}
    """
    from processing.pdf_processor import load_chunks
    try:
        all_chunks = load_chunks(business_name) or all_chunks
    except Exception:
        pass

    initial_state: AgentState = {
        "user_id":          user_id,
        "conversation_id":  conversation_id,
        "query":            query,
        "business_name":    business_name,
        "business_type":    business_type,
        "all_chunks":       all_chunks,
        "history":          [],
        "user_memory":      [],
        "current_cart":     [],
        "retrieved_chunks": [],
        "knowledge_text":   "",
        "customer_context": {},
        "tool_results":     {},
        "critic_ok":        True,
        "response":         "",
    }

    try:
        final_state = _AGENT_GRAPH.invoke(initial_state)
    except Exception as exc:
        print(f"[AGENT] Error: {exc}")
        import traceback
        traceback.print_exc()
        return {
            "response": "I'm sorry, something went wrong. Please try again.",
            "tool_results": {},
            "ics_file_url": "",
        }

    ics_url = ""
    for result in final_state.get("tool_results", {}).values():
        if isinstance(result, dict) and result.get("ics_file_url"):
            ics_url = result["ics_file_url"]
            break

    return {
        "response":     final_state.get("response", ""),
        "tool_results": final_state.get("tool_results", {}),
        "ics_file_url": ics_url,
    }