"""
agent/rag_engine.py
-------------------
LLM-only Retrieval-Augmented Generation (no embeddings, no vector DB).

Uses separate LLM calls for:
  - Chunking queries into logical topics
  - Chunking customer info into logical topics
  - Matching (topic, PDF-chunk) pairs to find relevant knowledge
  - Building the composite prompt for final response generation

All calls use llm_call() (structured, low-temp) not llm_chat().
"""

import os
import json
import sqlite3

from core.llm_client import llm_call
from core.logger import log_agent_event


# --------------------------------------------------------------------------
# Topic extraction
# --------------------------------------------------------------------------

def chunk_query_into_topics(
    query: str,
    history: list[dict] | None = None,
) -> list[str]:
    """
    LLM call I: divide the customer query into distinct logical search topics.
    Returns a list of short topic strings for RAG matching.
    """
    history_text = ""
    if history:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-4:]
        )

    prompt = f"""Divide the customer query below into distinct logical topics for knowledge base search.
Each topic must be a short phrase (3-8 words) that can be matched against a business knowledge base.
Extract ALL relevant topics -- ordering, scheduling, pricing, policies, services, complaints, etc.

Conversation context (last 4 turns):
{history_text or "None"}

Customer query: "{query}"

Return ONLY a JSON array of topic strings.
Example: ["pizza menu options", "delivery address", "order total price"]
No markdown. Just the JSON array."""

    raw = llm_call(prompt, temperature=0.0, max_tokens=200)
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            topics = json.loads(raw[start:end])
            return [str(t) for t in topics if t and isinstance(t, str)]
    except Exception:
        pass
    # Fallback: use the raw query as a single topic
    return [query]


def chunk_customer_info(customer_context: dict) -> list[dict]:
    """
    LLM call II: divide customer profile data into logical topic groups
    for context-aware recommendations (family, history, loyalty, etc.).
    """
    if not customer_context or all(not v for v in customer_context.values()):
        return []

    ctx_text = json.dumps(customer_context, indent=2, default=str)

    prompt = f"""Divide this customer's profile information into logical topic groups.
Each group should have a short label and the relevant data condensed to a single line.

Customer profile data:
{ctx_text}

Return ONLY a JSON array like:
[
  {{"topic": "Contact Info", "data": "Name: John Smith, Phone: 555-1234, Email: j@x.com"}},
  {{"topic": "Order History", "data": "3 past orders, favourite: Margherita Pizza"}},
  {{"topic": "Loyalty Status", "data": "450 points, Silver tier"}}
]
No markdown. Just the JSON array."""

    raw = llm_call(prompt, temperature=0.0, max_tokens=500)
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            result = json.loads(raw[start:end])
            return result if isinstance(result, list) else []
    except Exception:
        pass
    return [{"topic": "Full Profile", "data": ctx_text[:600]}]


# --------------------------------------------------------------------------
# Chunk retrieval
# --------------------------------------------------------------------------

def retrieve_relevant_chunks(
    topics: list[str],
    all_chunks: list[dict],
    top_n: int = 5,
    business_name: str = "",
) -> tuple[list[dict], list[int]]:
    """
    LLM call III: match (topic list, PDF chunks) pairs to select relevant chunks.
    Returns (selected_chunks, selected_indices).
    Processes all chunks in batches of 40.
    """
    if not all_chunks or not topics:
        return [], []

    batch_size = 40
    relevant_indices: set[int] = set()

    for batch_start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[batch_start: batch_start + batch_size]
        chunk_summaries = "\n".join(
            f"[{batch_start + i}] {c.get('section_title', 'Info')}: "
            f"{c.get('text', '')[:150].replace(chr(10), ' ')}"
            for i, c in enumerate(batch)
        )

        prompt = f"""You are selecting the most relevant knowledge chunks for a customer service query.

Search topics (what the customer needs):
{json.dumps(topics)}

Knowledge chunks (numbered):
{chunk_summaries}

Select the chunk indices that contain information relevant to ANY of the search topics.
Return ONLY a JSON array of integer indices.
Example: [0, 3, 7]
Return at most {top_n} indices. If nothing is relevant, return [].
No markdown. Just the JSON array."""

        raw = llm_call(prompt, temperature=0.0, max_tokens=150)
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                indices = json.loads(raw[start:end])
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(all_chunks):
                        relevant_indices.add(idx)
        except Exception:
            continue

    selected_indices = sorted(relevant_indices)[:top_n]
    retrieved = [all_chunks[i] for i in selected_indices]

    # Also pull matching enriched knowledge from DB (scoped by business)
    enriched = _fetch_enriched_knowledge(topics, business_name)
    for ek in enriched:
        retrieved.append({
            "text": ek["content"],
            "section_title": f"[Enriched] {ek['topic']}",
            "id": -1,
        })

    return retrieved, selected_indices


def _fetch_enriched_knowledge(topics: list[str], business_name: str = "") -> list[dict]:
    """Keyword-match topics against the enriched_knowledge table (from DB, scoped by business)."""
    db_name = os.getenv("DB_NAME", "business_agent.db")
    try:
        conn = sqlite3.connect(db_name, timeout=30.0)
        conn.row_factory = sqlite3.Row
        results: list[dict] = []
        seen_topics: set[str] = set()
        for topic in topics:
            for kw in topic.lower().split():
                if len(kw) < 3:
                    continue
                if business_name:
                    rows = conn.execute(
                        """SELECT topic, content FROM enriched_knowledge
                           WHERE (business_name = ? OR business_name IS NULL OR business_name = '')
                             AND (LOWER(topic) LIKE ? OR LOWER(content) LIKE ?)
                           LIMIT 2""",
                        (business_name, f"%{kw}%", f"%{kw}%"),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT topic, content FROM enriched_knowledge
                           WHERE LOWER(topic) LIKE ? OR LOWER(content) LIKE ?
                           LIMIT 2""",
                        (f"%{kw}%", f"%{kw}%"),
                    ).fetchall()
                for r in rows:
                    if r["topic"] not in seen_topics:
                        seen_topics.add(r["topic"])
                        results.append(dict(r))
        conn.close()
        return results[:5]
    except Exception:
        return []


# --------------------------------------------------------------------------
# Customer context loader
# --------------------------------------------------------------------------

def get_customer_context(
    user_id: int,
    session_id: str = "",
    business_name: str = "",
) -> dict:
    """Gather all customer data for context building."""
    db_name = os.getenv("DB_NAME", "business_agent.db")
    try:
        conn = sqlite3.connect(db_name, timeout=30.0)
        conn.row_factory = sqlite3.Row

        user = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not user:
            conn.close()
            return {}

        orders = conn.execute(
            """SELECT o.id, o.status, o.order_type, o.scheduled_at, o.total_price,
                      o.delivery_type, o.items_json, s.name AS service_name
               FROM orders o LEFT JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? AND s.business_name = ?
               ORDER BY o.scheduled_at DESC LIMIT 8""",
            (user_id, business_name),
        ).fetchall()

        loyalty = conn.execute(
            "SELECT points, tier FROM loyalty_points WHERE user_id = ? AND business_name = ?",
            (user_id, business_name),
        ).fetchone()

        fav = conn.execute(
            """SELECT s.name, COUNT(o.id) AS cnt FROM orders o
               JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? AND s.business_name = ?
               GROUP BY s.name ORDER BY cnt DESC LIMIT 1""",
            (user_id, business_name),
        ).fetchone()

        cart_items = conn.execute(
            """SELECT c.*, s.name AS service_name FROM cart c
               LEFT JOIN services s ON c.service_id = s.id
               WHERE c.user_id = ? AND c.session_id = ? AND c.business_name = ?""",
            (user_id, session_id, business_name),
        ).fetchall() if session_id else []

        complaints = conn.execute(
            """SELECT c.complaint_type, c.description, c.status, c.created_at
               FROM complaints c
               LEFT JOIN orders o ON c.order_id = o.id
               LEFT JOIN services s ON o.service_id = s.id
               WHERE c.user_id = ? AND (s.business_name = ? OR c.order_id IS NULL)
               ORDER BY c.created_at DESC LIMIT 3""",
            (user_id, business_name),
        ).fetchall()

        conn.close()

        family_raw = user["family_members"] or "[]"
        try:
            family = json.loads(family_raw)
        except Exception:
            family = []

        return {
            "user_id": user_id,
            "full_name": user["full_name"] or "",
            "email": user["email"] or "",
            "phone": user["phone"] or "",
            "address": user["address"] or "",
            "postal_code": user["postal_code"] or "",
            "city": user["city"] or "",
            "family_members": family,
            "loyalty_points": loyalty["points"] if loyalty else 0,
            "loyalty_tier": loyalty["tier"] if loyalty else "bronze",
            "usual_favorite": fav["name"] if fav else "",
            "usual_favorite_count": fav["cnt"] if fav else 0,
            "recent_orders": [dict(o) for o in orders],
            "recent_complaints": [dict(c) for c in complaints],
            "current_cart": [dict(c) for c in cart_items],
        }
    except Exception as e:
        print(f"[RAG] get_customer_context error: {e}")
        return {}


# --------------------------------------------------------------------------
# Composite prompt builder (used by agent.py for final LLM response call)
# --------------------------------------------------------------------------

def build_composite_prompt(
    query: str,
    retrieved_chunks: list[dict],
    customer_context: dict,
    tool_results: dict,
    conversation_history: list[dict],
    business_name: str,
    business_type: str,
    current_cart: list[dict] | None = None,
) -> str:
    """
    Build the final composite prompt passed to llm_chat() for response generation.
    Includes: knowledge base, customer profile, tool results, conversation history.
    """
    from datetime import datetime

    today_str = datetime.now().strftime("%Y-%m-%d (%A)")

    # ── Knowledge base ──────────────────────────────────────────────────────
    chunks_text = "\n\n".join(
        f"[{c.get('section_title', 'Info')}]\n{c.get('text', '')[:400]}"
        for c in retrieved_chunks
    ) or "No specific knowledge retrieved for this query."

    # ── Customer profile (topic-chunked) ───────────────────────────────────
    customer_topics = chunk_customer_info(customer_context)
    if customer_topics:
        customer_text = "\n".join(
            f"  {t.get('topic', 'Info')}: {t.get('data', '')}"
            for t in customer_topics
        )
    else:
        customer_text = "No customer profile available."

    # ── Current cart ───────────────────────────────────────────────────────
    cart_text = ""
    if current_cart:
        total = round(sum(
            int(i.get("qty", 1)) * float(i.get("price", 0.0))
            for i in current_cart
        ), 2)
        cart_lines = "\n".join(
            f"  - {int(i.get('qty', 1))}x {i.get('name', 'Item')} @ ${float(i.get('price', 0.0)):.2f}"
            f"{' (' + str(i.get('modifiers', '')) + ')' if i.get('modifiers') else ''}"
            for i in current_cart
        )
        cart_text = f"CURRENT CART:\n{cart_lines}\n  Subtotal: ${total:.2f} (EXACT — use this value, never recalculate)"

    # ── Tool results ───────────────────────────────────────────────────────
    tools_text = ""
    if tool_results:
        for tool_name, result in tool_results.items():
            status = "SUCCESS" if result.get("success") else "FAILED"
            tools_text += f"\n--- {tool_name} [{status}] ---\n"
            # Include the message and key fields, skip raw DB dumps
            tools_text += f"Message: {result.get('message', '')}\n"
            for key in ("order_ref", "booking_ref", "total", "date", "time", "provider",
                        "items", "cart_items", "cart_total", "address", "points_earned",
                        "recent_order_cost_breakdown", "summary",
                        "needs_confirmation", "pending_booking",
                        "error", "needs_info", "missing_fields", "conflict"):
                if key in result and result[key] not in (None, "", [], {}):
                    tools_text += f"{key}: {result[key]}\n"

    # ── Conversation history ───────────────────────────────────────────────
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:600]}"
        for m in (conversation_history or [])[-12:]
    ) or "No prior messages."

    # ── Complaint context ──────────────────────────────────────────────────
    complaints = customer_context.get("recent_complaints", [])
    complaints_text = ""
    if complaints:
        complaints_text = "PAST COMPLAINTS:\n" + "\n".join(
            f"  [{c['created_at'][:10]}] {c['complaint_type']}: {c['description']} "
            f"(Status: {c['status']})"
            for c in complaints
        )

    # ── Favourite / upsell reminder ────────────────────────────────────────
    fav = customer_context.get("usual_favorite", "")
    fav_note = (
        f"\nNOTE: This customer's usual favourite is '{fav}'. "
        "Mention it naturally if relevant (e.g. 'Would you like your usual {fav}?')."
    ) if fav else ""

    # ── Family cross-sell ──────────────────────────────────────────────────
    family = customer_context.get("family_members", [])
    family_note = ""
    if family:
        names = ", ".join(
            f"{m.get('name', '')} ({m.get('relation', '')})" for m in family
        )
        family_note = (
            f"\nFAMILY ON FILE: {names}. "
            "Suggest services for family members when it naturally fits the conversation."
        )

    prompt = f"""You are the AI customer service agent for "{business_name}" ({business_type}).
Today's date: {today_str}

=== KNOWLEDGE BASE ===
{chunks_text}

=== CUSTOMER PROFILE ===
{customer_text}
{complaints_text}

=== {cart_text or 'CART: Empty'}

=== TOOL RESULTS ===
{tools_text.strip() or 'No tool was called this turn.'}

=== CONVERSATION HISTORY ===
{history_text}

=== CURRENT CUSTOMER MESSAGE ===
{query}
{fav_note}{family_note}

RESPONSE INSTRUCTIONS:
1. TONE: Be warm, friendly, and conversational — like a knowledgeable human assistant, not a form.
   Use the customer's name naturally when you know it. Add light friendly phrases where appropriate.
2. GROUNDING: Only state facts from the KNOWLEDGE BASE, TOOL RESULTS, or CUSTOMER PROFILE.
   Never invent prices, dates, availability, staff names, or policies.
3. PENDING CONFIRMATION: If a tool returned needs_confirmation=true, the slot/booking was VALIDATED.
   Present the summary from the tool and ask "Would you like me to confirm?" Do NOT say it was booked yet.
   CRITICAL: Do NOT apply 24h/advance-booking policies here — the tool already validated the slot.
   If the tool says "Great news — that slot is available", the slot IS available. Ask for confirmation only.
4. TOOL SUCCESS: When success=true, the action is DONE. State the outcome clearly. Do NOT ask for
   confirmation — the tool already completed the action. Examples: remove_from_cart cleared the cart
   → say "Your cart has been cleared"; confirm_order placed the order → say "Your order is confirmed".
   Never ask "would you like me to clear it?" or "shall I proceed?" when the tool already succeeded.
5. TOOL FAILURE: If success=false AND needs_confirmation is NOT true, explain the specific reason warmly.
   If needs_confirmation=true, treat it as rule 3 — present the summary and ask for confirmation.
   Never bypass or ignore the failure message when it is a real failure (conflict, missing info, etc.).
6. ORDER HISTORY: Show the actual booking records from the tool. If total=0, say warmly that they
   have no previous bookings yet — never hallucinate a number.
7. NEVER confirm an order or booking unless a tool returned success:true with a reference number.
8. CART & ORDER: TOOL RESULTS and CURRENT CART are AUTHORITATIVE. If view_cart says "cart is empty"
   or confirm_order just succeeded, NEVER list items from conversation history. If the tool says empty,
   say empty. If the tool shows items A+B, show A+B — never substitute with what the user "wanted" (e.g.
   user wanted Lasagna but cart has Pasta Carbonara → show Pasta Carbonara, the actual cart).
   When confirm_order returns success: Show ONLY the "items" and "total" from the tool. The tool's items
   are the actual confirmed order. Do NOT add any items from conversation history (e.g. if user wanted
   Gelato+Lasagna but the order had only Garlic Bread, show only Garlic Bread and the tool's total).
   For "cost breakdown" / "my order total" after order placed: use recent_order_cost_breakdown from
   get_order_history — NEVER use get_pricing or conversation history for the user's actual order.
9. CLARIFICATION: Ask ONE question at a time. Don't bombard the customer.
10. NAMES: Use the customer's real name. Never use [Name] or [Customer].
11. HOLIDAYS: Only call a date a holiday if it literally is that calendar date.
12. FUTURE DATES: Never refuse a future appointment. Businesses book in advance.
13. UPSELL: Suggest ONE related item per session naturally ("Would you like X with that?").
14. HISTORY CONTINUITY: Reference what the customer said earlier if relevant.
15. LENGTH: Keep responses friendly but concise. No filler phrases like "Certainly!" or "Of course!".

Now respond to the customer:"""

    return prompt