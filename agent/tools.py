"""
agent/tools.py
--------------
All agent tools. Each tool:
  - Has a clear TOOL_DESCRIPTION used by the LLM router
  - Accepts (state: dict, **kwargs) -> dict
  - Returns a result dict
  - Is fully generic (no hardcoded business logic)

Tools include: cart management, booking, scheduling, calendar,
address validation, recommendations, loyalty, dispute handling, etc.
"""

import os
import json
import sqlite3
import random
import re
from datetime import datetime, timedelta, timezone

from core.database import get_business_meta, get_global_stats
from core.logger import log_agent_event, log_db_write
from core.llm_client import llm_call

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False



def _find_next_available_slots(
    conn,
    provider_id: int,
    after_dt,
    duration_min: int,
    count: int = 3,
) -> list:
    """
    Find the next N available time slots for a provider after a given datetime,
    skipping any slots that conflict with existing confirmed/pending bookings.
    Looks up to 14 days ahead, checks only business hours (9:00-17:00).
    """
    from datetime import datetime, timedelta

    slots = []
    check_dt = after_dt + timedelta(hours=1)  # start looking 1 hour after requested time
    # Round up to next full hour
    check_dt = check_dt.replace(minute=0, second=0, microsecond=0)

    # Get provider schedule
    prov = conn.execute(
        "SELECT schedule FROM service_providers WHERE id = ?", (provider_id,)
    ).fetchone()
    schedule = {}
    if prov and prov["schedule"]:
        try:
            import json as _json
            schedule = _json.loads(prov["schedule"])
        except Exception:
            schedule = {}

    days_checked = 0
    while len(slots) < count and days_checked < 14:
        day_key = check_dt.strftime("%A")[:3].lower()  # mon, tue, wed...

        # Skip if provider not available this day
        if schedule and day_key not in schedule:
            check_dt = (check_dt + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
            days_checked += 1
            continue

        # Only check business hours 9:00-17:00
        if check_dt.hour < 9:
            check_dt = check_dt.replace(hour=9, minute=0, second=0, microsecond=0)
        elif check_dt.hour >= 17:
            check_dt = (check_dt + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
            days_checked += 1
            continue

        slot_end = check_dt + timedelta(minutes=duration_min)

        # Check for conflict
        conflict = conn.execute(
            """SELECT id FROM orders
               WHERE provider_id = ?
                 AND status IN ('pending','confirmed')
                 AND datetime(scheduled_at) < datetime(?)
                 AND datetime(scheduled_at, '+' || COALESCE(
                     (SELECT duration_min FROM services WHERE id = orders.service_id), 30
                   ) || ' minutes') > datetime(?)
               LIMIT 1""",
            (provider_id, slot_end.isoformat(), check_dt.isoformat()),
        ).fetchone()

        if not conflict:
            slots.append(check_dt)

        check_dt = check_dt + timedelta(hours=1)

    return slots

def _db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False, timeout=60.0)
    conn.row_factory = sqlite3.Row
    # WAL mode allows concurrent readers + 1 writer — eliminates most lock errors on Windows
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")   # 60-second busy wait in ms
    return conn


def _resolve_service_name(conn, service_id: int) -> str:
    """Helper to get a human-readable service name for any ID."""
    if not service_id:
        return "Unknown Service"
    row = conn.execute("SELECT name FROM services WHERE id = ?", (service_id,)).fetchone()
    return row["name"] if row else f"Service {service_id}"


def _generate_ref(prefix: str, db_id: int) -> str:
    """Generate a customer-facing reference number."""
    return f"{prefix}-{str(db_id).zfill(5)}"


# ─────────────────────────────────────────────
# Cart / Order Management Tools
# ─────────────────────────────────────────────

def add_to_cart(state: dict, **kwargs) -> dict:
    """Add an item to the customer's shopping cart."""
    user_id = state.get("user_id")
    query = state.get("query", "")
    session_id = state.get("conversation_id", "")
    conn = _db()
    try:
        services = conn.execute("SELECT * FROM services WHERE business_name = ? ORDER BY id", (state.get("business_name"),)).fetchall()
        if not services:
            return {"success": False, "message": "No services or items available."}

        # Get last agent message for anaphora resolution ("yes" -> items agent offered to add)
        history = state.get("history", [])
        last_agent = ""
        for m in reversed(history):
            if m.get("role") == "assistant":
                last_agent = (m.get("content") or "")[:600]
                break

        current_cart = state.get("current_cart", [])
        cart_names = {str(c.get("name", "")).lower() for c in current_cart}
        cart_note = f'\nAlready in cart (do NOT add again): {", ".join(cart_names)}' if cart_names else ""

        svc_list = "\n".join(
            f"- ID:{s['id']} | {s['name']} | ${s['price']:.2f} | {s['category'] or 'General'} | Modifiers: {s['modifiers'] or 'none'}"
            for s in services
        )
        context_note = f'\nLast agent message: "{last_agent}"' if last_agent else ""
        match_prompt = f"""Customer request: "{query}"
Available services:
{svc_list}
{context_note}
{cart_note}

Extract ALL items to add. Rules:
1. If customer lists items (e.g. "X, Y and Z") -> return ALL with correct service_id.
2. CRITICAL for "yes"/"add that"/"sure": When the agent offered specific items (e.g. "Shall I add both the Gelato and Lasagna?" or "add Gelato and Lasagna to your cart?"), return ONLY those offered items. "That" = the items the agent explicitly offered. Do NOT return items already in cart.
3. Do NOT add items already in cart when resolving "yes".
4. NEVER invent items. Only items from customer message OR last agent message.
5. If nothing matchable: {{"items": [{{"matched": false, "message": "What would you like to add?"}}]}}

Return ONLY JSON: {{"items": [{{"service_id": N, "quantity": 1, "modifiers": "", "matched": true}}, ...]}}
No markdown."""

        raw = llm_call(match_prompt, temperature=0.2, max_tokens=400)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            match_data = json.loads(raw)
            items_to_add = match_data.get("items", [])
        except json.JSONDecodeError:
            items_to_add = []

        if not items_to_add:
            return {
                "success": False,
                "needs_info": True,
                "message": (
                    "What would you like to add? Please type the item name, "
                    "for example: **Margherita Large** or **Pepperoni Medium**"
                ),
            }

        # Build name->service map for validation (avoid Lasagna->Pasta Carbonara substitution)
        combined_text = f"{query} {last_agent}".lower()
        name_to_svc = {}
        for s in services:
            n = (s["name"] or "").lower()
            if len(n) > 2:
                name_to_svc[n] = dict(s)
                for word in n.replace("-", " ").split():
                    if len(word) > 3 and word not in ("with", "and", "the", "for"):
                        name_to_svc[word] = dict(s)

        added_log = []
        first_unmatched_msg = None
        for item in items_to_add:
            if not item.get("matched", False):
                if first_unmatched_msg is None and item.get("message"):
                    first_unmatched_msg = item["message"]
                continue

            svc_id = item.get("service_id")
            quantity = max(1, int(item.get("quantity", 1)))
            modifiers = item.get("modifiers", "")

            svc = conn.execute("SELECT * FROM services WHERE id = ? AND business_name = ?", (svc_id, state.get("business_name"))).fetchone()
            if not svc:
                continue

            # Validate: if user said "Lasagna" but LLM returned Pasta Carbonara, prefer Lasagna
            svc_name_lower = (svc["name"] or "").lower()
            if svc_name_lower not in combined_text:
                for token in combined_text.replace(",", " ").replace(" and ", " ").split():
                    token = token.strip()
                    if len(token) > 4 and token in name_to_svc:
                        alt = name_to_svc[token]
                        if str(alt.get("business_name", "")) == str(state.get("business_name", "")):
                            svc = alt
                            svc_id = svc["id"]
                            break

            cur_cart = conn.execute(
                "INSERT INTO cart (user_id, session_id, business_name, service_id, service_name, quantity, unit_price, modifiers, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, session_id, state.get("business_name"), svc_id, svc["name"], quantity, svc["price"], modifiers, ""),
            )
            insert_id = cur_cart.lastrowid
            added_log.append(f"{quantity}x {svc['name']}")

        if not added_log:
            msg = first_unmatched_msg or "I could not find that item. Please type the exact item name from the menu."
            return {"success": False, "needs_info": True, "message": msg}

        conn.commit()
        # Log after commit to avoid DB lock contention
        try:
            log_db_write(state.get("conversation_id", ""), "cart", "INSERT_BATCH", None, {"added": added_log})
        except Exception:
            pass

        # Get current cart summary
        cart_items = conn.execute(
            "SELECT * FROM cart WHERE user_id = ? AND session_id = ? AND business_name = ?",
            (user_id, session_id, state.get("business_name", "")),
        ).fetchall()
        total = sum(item["quantity"] * item["unit_price"] for item in cart_items)

        return {
            "success": True,
            "added_items": added_log,
            "cart_items": [
                {"name": c["service_name"], "qty": c["quantity"], "price": c["unit_price"], "modifiers": c["modifiers"]}
                for c in cart_items
            ],
            "cart_total": round(total, 2),
            "message": f"Added {', '.join(added_log)} to your cart. Current total: ${total:.2f}",
        }
    finally:
        conn.close()


def remove_from_cart(state: dict, **kwargs) -> dict:
    """Remove an item from the customer's cart or clear the entire cart."""
    user_id = state.get("user_id")
    query = state.get("query", "")
    session_id = state.get("conversation_id", "")
    conn = _db()
    try:
        cart_items = conn.execute(
            "SELECT * FROM cart WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchall()

        if not cart_items:
            return {"success": False, "message": "Your cart is empty."}

        # Check if user wants to clear all
        q_lower = query.lower()
        if any(w in q_lower for w in ["clear", "empty", "remove all", "start over"]):
            conn.execute(
                "DELETE FROM cart WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            conn.commit()
            try:
                log_db_write(state.get("conversation_id", ""), "cart", "DELETE_ALL", None, {"user_id": user_id, "session_id": session_id})
            except Exception:
                pass
            return {"success": True, "message": "Cart cleared. What would you like to add?", "cart_items": [], "cart_total": 0}

        # Use LLM to figure out which item to remove (include last agent for "yes" after "replace X with Y")
        history = state.get("history", [])
        last_agent = ""
        for m in reversed(history):
            if m.get("role") == "assistant":
                last_agent = (m.get("content") or "")[:400]
                break
        items_str = "\n".join(
            f"- CartID:{c['id']} | {c['service_name']} | qty:{c['quantity']}"
            for c in cart_items
        )
        context_note = f'\nLast agent message (use when customer said "yes" to replace/correct): "{last_agent}"' if last_agent else ""
        match_prompt = f"""Customer says: "{query}"
Items in cart:
{items_str}
{context_note}

Which item should be removed? If agent offered "replace X with Y" and customer said "yes", remove X. Return ONLY JSON: {{"cart_id": N, "found": true}}
If unclear: {{"found": false, "message": "which item?"}}
No markdown, just JSON."""

        raw = llm_call(match_prompt, temperature=0.1, max_tokens=150)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            match = json.loads(raw)
        except json.JSONDecodeError:
            match = {"found": False, "message": "Could not determine which item to remove."}

        if not match.get("found", False):
            return {"success": False, "message": match.get("message", "Which item would you like to remove?")}

        cart_id = match.get("cart_id")
        removed = conn.execute("SELECT service_name FROM cart WHERE id = ?", (cart_id,)).fetchone()
        conn.execute("DELETE FROM cart WHERE id = ?", (cart_id,))
        log_db_write(state.get("conversation_id", ""), "cart", "DELETE", cart_id, {"service": removed["service_name"] if removed else "unknown"})
        conn.commit()

        remaining = conn.execute(
            "SELECT * FROM cart WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchall()
        total = sum(item["quantity"] * item["unit_price"] for item in remaining)

        return {
            "success": True,
            "removed_item": removed["service_name"] if removed else "item",
            "cart_items": [
                {"name": c["service_name"], "qty": c["quantity"], "price": c["unit_price"], "modifiers": c["modifiers"]}
                for c in remaining
            ],
            "cart_total": round(total, 2),
            "message": f"Removed {removed['service_name'] if removed else 'item'} from your cart. Total: ${total:.2f}",
        }
    finally:
        conn.close()


def view_cart(state: dict, **kwargs) -> dict:
    """Show the customer's current cart contents."""
    user_id = state.get("user_id")
    session_id = state.get("conversation_id", "")
    business_name = state.get("business_name", "")
    conn = _db()
    try:
        cart_items = conn.execute(
            """SELECT c.*, s.business_name as svc_business FROM cart c
               LEFT JOIN services s ON c.service_id = s.id
               WHERE c.user_id = ? AND c.session_id = ? AND c.business_name = ?""",
            (user_id, session_id, business_name),
        ).fetchall()
        if not cart_items:
            # Check if there's a very recent confirmed order for this session/business
            recent_order = conn.execute(
                """SELECT o.id, o.total_price, o.items_json FROM orders o
                   WHERE o.user_id = ? AND o.status = 'confirmed' AND o.business_name = ?
                   ORDER BY o.id DESC LIMIT 1""",
                (user_id, business_name),
            ).fetchone()
            if recent_order:
                order_ref = _generate_ref("ORDER", recent_order["id"])
                return {
                    "success": True,
                    "message": f"Your cart is empty — your last order ({order_ref}) was already confirmed and placed. Total: ${recent_order['total_price']:.2f}. Would you like to place a new order?",
                    "cart_items": [],
                    "cart_total": 0,
                    "recent_order_ref": order_ref,
                }
            return {"success": True, "message": "Your cart is empty. What would you like to add?", "cart_items": [], "cart_total": 0}

        total = sum(item["quantity"] * item["unit_price"] for item in cart_items)
        return {
            "success": True,
            "cart_items": [
                {"name": c["service_name"], "qty": c["quantity"], "price": c["unit_price"], "modifiers": c["modifiers"]}
                for c in cart_items
            ],
            "cart_total": round(total, 2),
            "message": f"Your cart has {len(cart_items)} item(s) totaling ${total:.2f}.",
        }
    finally:
        conn.close()


def confirm_order(state: dict, **kwargs) -> dict:
    """Confirm and finalise the cart as a completed order."""
    # Safety guard: confirm_order must never fire for pure appointment businesses
    biz_type = state.get("business_type", "")
    APPT_KEYWORDS = ["dental","dentist","clinic","doctor","medical","salon","beauty",
                     "spa","massage","therapy","physiotherapy","photography","photographer",
                     "gym","fitness","yoga","barber","barbershop","dermatology","nail","lash"]
    if any(k in biz_type.lower() for k in APPT_KEYWORDS):
        return {
            "success": False,
            "needs_booking": True,
            "message": "This business uses appointment booking. Please use the booking flow to schedule your service.",
        }

    user_id = state.get("user_id")
    session_id = state.get("conversation_id", "")
    conn = _db()
    try:
        # Check if cart exists for THIS session, user, and business
        business_name = state.get("business_name", "")
        cart_items = conn.execute(
            "SELECT * FROM cart WHERE user_id = ? AND session_id = ? AND business_name = ?",
            (user_id, session_id, business_name),
        ).fetchall()

        if not cart_items:
            # Check if there are ANY items in cart for this user (could be different session)
            any_items = conn.execute("SELECT COUNT(*) FROM cart WHERE user_id = ?", (user_id,)).fetchone()[0]
            if any_items > 0:
                return {
                    "success": False,
                    "message": "I found items in your cart from a different session, but your current session is empty. Please verify your order."
                }
            return {"success": False, "needs_info": True, "message": "Your cart is empty. Please add items first by typing their name, e.g. **Margherita Large** or **Pepperoni Medium**."}

        # Calculate total from the SAME cart_items we're ordering (ensures consistency)
        total = round(sum(float(c["quantity"] or 1) * float(c["unit_price"] or 0.0) for c in cart_items), 2)

        items_json = json.dumps([
            {"name": c["service_name"], "qty": c["quantity"], "price": c["unit_price"], "modifiers": c["modifiers"]}
            for c in cart_items
        ])

        # Create the order
        cur = conn.execute(
            """INSERT INTO orders (business_name, user_id, service_id, status, order_type, scheduled_at, total_price, items_json, notes)
               VALUES (?, ?, ?, 'confirmed', 'order', ?, ?, ?, ?)""",
            (
                business_name,
                user_id,
                cart_items[0]["service_id"],
                datetime.now(timezone.utc).isoformat(),
                total,
                items_json,
                f"Order confirmed via AI assistant for {business_name}",
            ),
        )
        order_id = cur.lastrowid

        # Clear cart
        conn.execute(
            "DELETE FROM cart WHERE user_id = ? AND session_id = ? AND business_name = ?",
            (user_id, session_id, business_name),
        )

        # Award loyalty points
        points_earned = int(total * 10)
        curr = conn.execute("SELECT points FROM loyalty_points WHERE user_id = ?", (user_id,)).fetchone()
        new_points = (curr["points"] if curr else 0) + points_earned
        new_tier = "gold" if new_points >= 1000 else "silver" if new_points >= 500 else "bronze"
        
        if curr:
            conn.execute(
                """UPDATE loyalty_points SET points = ?, tier = ?, updated_at = ?
                   WHERE user_id = ?""",
                (new_points, new_tier, datetime.now(timezone.utc).isoformat(), user_id),
            )
        else:
            conn.execute(
                """INSERT INTO loyalty_points (user_id, points, tier, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (user_id, new_points, new_tier, datetime.now(timezone.utc).isoformat()),
            )
        order_ref = _generate_ref("ORDER", order_id)

        # ── ICS & Calendar Integration ──
        ics_filename = None
        for item in cart_items:
            if any(bt in state.get("business_type", "").lower() for bt in ["photography", "dental", "cleaner", "salon", "clinic"]):
                start_dt = datetime.now() + timedelta(days=1)
                end_dt = start_dt + timedelta(minutes=60)
                conn.execute(
                    """INSERT INTO calendar_events (business_name, user_id, order_id, title, description, start_time, end_time, provider)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (state.get("business_name"), user_id, order_id, item["service_name"], f"Order {order_ref}", start_dt.isoformat(), end_dt.isoformat(), "Assigned Staff")
                )
                ics_path = _generate_ics(order_id, item["service_name"], start_dt, end_dt, "Assigned Staff", state.get("customer_context", {}).get("full_name", "Customer"))
                ics_filename = os.path.basename(ics_path)
                break
        conn.commit()
        conn.close()
        conn = None
        # Audit logging after commit to avoid DB lock
        try:
            log_db_write(state.get("conversation_id", ""), "orders", "INSERT", order_id, {"total": total, "status": "confirmed"})
            log_db_write(state.get("conversation_id", ""), "cart", "DELETE_ALL", None, {"reason": "order_confirmed", "order_id": order_id})
            log_db_write(state.get("conversation_id", ""), "loyalty_points", "UPDATE" if curr else "INSERT", None, {"user_id": user_id, "points": new_points, "tier": new_tier})
        except Exception:
            pass

        res = {
            "success": True,
            "order_id": order_id,
            "order_ref": order_ref,
            "items": json.loads(items_json),
            "total": round(total, 2),
            "points_earned": points_earned,
            "cart_items": [],
            "cart_total": 0,
            "message": f" Order confirmed! Your reference number is {order_ref}. Total: ${total:.2f}. You earned {points_earned} loyalty points.",
        }
        if ics_filename:
            res["ics_file_url"] = f"/calendar/{ics_filename}"
            res["message"] += f" A calendar event has been created: {ics_filename}"
        
        return res
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ─────────────────────────────────────────────
# Booking / Scheduling Tools
# ─────────────────────────────────────────────

def _resolve_date_python(text: str, today: datetime | None = None) -> str | None:
    """
    Resolve a date expression to YYYY-MM-DD using pure Python — no LLM.
    Handles: "monday", "next monday", "this friday", "tomorrow", "today",
             "march 23", "23rd", "2026-03-23", etc.
    Returns None if cannot resolve.
    """
    if not text:
        return None
    today = today or datetime.now()
    t = text.lower().strip()

    # Already a date string
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(t, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Relative
    if t in ("today",):
        return today.strftime("%Y-%m-%d")
    if t in ("tomorrow",):
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    if t in ("yesterday",):
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")

    # Day of week — "monday", "next monday", "this monday"
    DAY_MAP = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3,
        "fri": 4, "sat": 5, "sun": 6,
    }
    is_next = "next" in t
    is_this = "this" in t

    for day_name, target_wd in DAY_MAP.items():
        if day_name not in t:
            continue
        today_wd = today.weekday()   # 0=Monday
        diff = (target_wd - today_wd) % 7

        if diff == 0:
            # Same weekday today
            if is_next:
                diff = 7   # "next monday" when today IS monday → 7 days
            else:
                diff = 7   # "monday" when today is monday → treat as next week
        elif is_next and diff <= 7:
            # "next monday" when monday is 3 days away → still 3 days (it IS next monday)
            pass
        # If not next/this and diff > 0, use the upcoming occurrence
        result = today + timedelta(days=diff)
        return result.strftime("%Y-%m-%d")

    # Try month + day: "march 23", "23rd march", "23 march"
    import re as _re
    MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    for mon_name, mon_num in MONTHS.items():
        if mon_name in t:
            nums = _re.findall(r"\d+", t)
            if nums:
                day_num = int(nums[0])
                year    = today.year
                try:
                    candidate = datetime(year, mon_num, day_num)
                    if candidate < today:
                        candidate = datetime(year + 1, mon_num, day_num)
                    return candidate.strftime("%Y-%m-%d")
                except ValueError:
                    pass

    return None


def book_appointment(state: dict, **kwargs) -> dict:
    """
    Book an appointment.  Two-phase flow:
      Phase 1 — parse slot, check conflict, save PENDING to conversation_state.
                Return needs_confirmation=True so agent asks the customer to confirm.
      Phase 2 — user said 'yes/confirm/book it', pending state found → commit to DB.
    """
    # Safety guard: never fire for pure ordering businesses (restaurants, pizzerias)
    biz_type = state.get("business_type", "")
    ORDER_KEYWORDS = ["restaurant","pizzeria","pizza","food delivery","fast food",
                      "takeout","takeaway","cafe","bakery","diner","catering"]
    if any(k in biz_type.lower() for k in ORDER_KEYWORDS):
        return {
            "success": False,
            "needs_order": True,
            "message": (
                "We're a restaurant — we take food orders for pickup or delivery  "
                "I'd be happy to help you place an order! What would you like? "
                "For table reservations, please call us directly."
            ),
        }

    user_id = state.get("user_id")
    query = state.get("query", "").lower()
    conversation_id = state.get("conversation_id", "")

    # ── Phase 2 check: is the user confirming a pending booking? ────────────
    from core.database import load_conversation_state, save_conversation_state
    conv_state = load_conversation_state(conversation_id) or {}
    pending = conv_state.get("pending_booking")

    is_confirmation = pending and any(w in query for w in [
        "yes", "confirm", "book it", "go ahead", "sure", "ok", "okay",
        "correct", "that's right", "sounds good", "please", "do it",
    ])

    if is_confirmation and pending:
        # ── COMMIT the pending booking ──────────────────────────────────
        conn = _db()
        try:
            user_row = conn.execute("SELECT full_name, username FROM users WHERE id = ?", (user_id,)).fetchone()
            patient_name = (
                ((user_row["full_name"] or "").strip() if user_row else "") or
                (user_row["username"] if user_row else "") or "Customer"
            )
            svc_id   = pending["service_id"]
            prov_id  = pending["provider_id"]
            req_dt   = datetime.fromisoformat(pending["scheduled_dt"])
            svc_name = pending["service_name"]
            price    = pending["price"]
            duration = pending["duration"]

            cur = conn.execute(
                """INSERT INTO orders (business_name, user_id, service_id, provider_id, status, order_type,
                                      scheduled_at, total_price, notes)
                   VALUES (?, ?, ?, ?, 'confirmed', 'appointment', ?, ?, ?)""",
                (state.get("business_name"), user_id, svc_id, prov_id, req_dt.isoformat(), price,
                 f"Booked via AI assistant for {state.get('business_name')}"),
            )
            order_id = cur.lastrowid
            ref      = _generate_ref("APPT", order_id)
            end_dt   = req_dt + timedelta(minutes=duration)

            conn.execute(
                """INSERT INTO calendar_events
                   (business_name, user_id, order_id, title, description, start_time, end_time, provider, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (state.get("business_name"), user_id, order_id, f"Appointment: {svc_name}",
                 f"Booking #{order_id}. Ref: {ref}",
                 req_dt.isoformat(), end_dt.isoformat(),
                 pending["provider_name"], "confirmed"),
            )
            conn.commit()
            conn.close()
            conn = None

            # Generate .ics
            _generate_ics(
                order_id, svc_name, req_dt, end_dt,
                pending["provider_name"], patient_name,
                business_name=state.get("business_name", ""),
                method="REQUEST", sequence=0,
            )

            # Clear pending state
            conv_state.pop("pending_booking", None)
            save_conversation_state(conversation_id, conv_state)

            safe_name   = re.sub(r"[^a-z0-9_]", "_", patient_name.lower())
            ics_filename = f"{safe_name}_appointment_{order_id}.ics"

            try:
                log_db_write(conversation_id, "orders", "INSERT", order_id,
                             {"type": "appointment", "service": svc_name,
                              "date": req_dt.isoformat()})
            except Exception:
                pass

            dt_friendly = req_dt.strftime("%A, %B %d, %Y at %I:%M %p").lstrip("0")
            return {
                "success": True,
                "booking_ref": ref,
                "booking_id": order_id,
                "service": svc_name,
                "provider": pending["provider_name"],
                "date": req_dt.strftime("%Y-%m-%d"),
                "time": req_dt.strftime("%H:%M"),
                "duration_min": duration,
                "price": price,
                "ics_file_url": f"/calendar/{ics_filename}",
                "message": (
                    f"Your appointment has been confirmed!\n"
                    f"  Service:   {svc_name}\n"
                    f"  Date/Time: {dt_friendly}\n"
                    f"  Provider:  {pending['provider_name']}\n"
                    f"  Duration:  {duration} min\n"
                    f"  Price:     ${price:.2f}\n"
                    f"  Reference: {ref}\n"
                    f"A calendar invitation has been saved. See you then!"
                ),
            }
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    # ── Phase 1: Parse the request and PROPOSE (don't commit yet) ──────────
    history = state.get("history", [])
    conn = _db()
    try:
        services  = conn.execute("SELECT * FROM services WHERE business_name = ? ORDER BY id",
                                 (state.get("business_name"),)).fetchall()
        providers = conn.execute(
            "SELECT * FROM service_providers WHERE available = 1 AND business_name = ? ORDER BY rating DESC",
            (state.get("business_name"),)
        ).fetchall()

        if not services:
            return {"success": False, "message": "No services are currently available for booking."}

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in (history or [])[-6:]
        )
        svc_list  = "\n".join(f"ID:{s['id']}|{s['name']}|${s['price']}|{s['duration_min']}min" for s in services)
        prov_list = "\n".join(f"ID:{p['id']}|{p['name']}|{p['specialty']}" for p in providers) if providers else "No providers listed"

        parse_prompt = (
            "Extract booking details from this conversation.\n\n"
            f"Conversation:\n{history_text}\nCUSTOMER: {state.get('query','')}\n\n"
            f"Available services:\n{svc_list}\n\n"
            f"Available providers:\n{prov_list}\n\n"
            f"Today's date: {datetime.now().strftime('%Y-%m-%d %A')}\n\n"
            "RULES:\n"
            "- Only set date if customer EXPLICITLY stated one (today/tomorrow/Monday/March 19 etc).\n"
            "- Only set time if customer EXPLICITLY stated one (2pm/14:00/morning etc).\n"
            "- Do NOT guess or infer date or time.\n"
            "- If the customer gave a day-of-week (e.g. Thursday), compute the actual YYYY-MM-DD.\n\n"
            "Return ONLY JSON:\n"
            '{"services": [{"service_id": 1, "service_name": "Dental Cleaning"}],\n'
            ' "date": "2026-03-19", "time": "14:00",\n'
            ' "provider_id": null, "parsed_ok": true,\n'
            ' "missing_fields": []}\n\n'
            "If date or time missing: set parsed_ok=false and list them in missing_fields.\n"
            "No markdown. Just JSON."
        )

        raw = llm_call(parse_prompt, temperature=0.0, max_tokens=400)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            booking = json.loads(raw)
        except json.JSONDecodeError:
            return {"success": False, "needs_info": True, "message": ("Please tell me the service, date and time. Example: **Dental Cleaning on Monday at 2pm**")}

        missing = booking.get("missing_fields", [])
        if not booking.get("parsed_ok", False) or missing:
            parts = []
            if "date" in missing:
                parts.append("the **date** (e.g. **Monday**, **March 25**, **tomorrow**)")
            if "time" in missing:
                parts.append("the **time** (e.g. **10am**, **2:30pm**, **14:00**)")
            if "service" in missing:
                parts.append("the **service** you'd like")
            if not parts:
                parts = ["the **date and time**"]
            ask = " and ".join(parts)
            return {
                "success": False,
                "needs_info": True,
                "missing_fields": missing,
                "message": (
                    f"To complete your booking I need {ask}. "
                    f"For example: **Dental Cleaning on Monday at 2pm**"
                ),
            }

        date_str = booking.get("date", "")
        time_str = booking.get("time", "")[:5]
        if not date_str or not time_str:
            return {
                "success": False,
                "needs_info": True,
                "message": (
                    "I need a specific date and time to book. "
                    "Please reply with the date and time, e.g. **Monday at 2pm** or **March 25 at 10:00**."
                ),
            }

        # ── Python date correction ────────────────────────────────────────────
        # The function-calling LLM sometimes mis-resolves day names.
        # We validate by checking the day-of-week; if wrong, re-resolve via Python.
        full_query = state.get("query", "") + " " + " ".join(
            m.get("content", "") for m in (state.get("history") or [])[-4:]
            if m.get("role") == "user"
        )
        today_dt = datetime.now()
        DAY_NAMES = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        mentioned_day = next((d for d in DAY_NAMES if d in full_query.lower()), None)

        if mentioned_day and date_str:
            try:
                resolved_dt  = datetime.fromisoformat(date_str)
                resolved_wd  = resolved_dt.strftime("%A").lower()
                if resolved_wd != mentioned_day:
                    # LLM got the wrong weekday — use Python resolver instead
                    corrected = _resolve_date_python(
                        ("next " if "next" in full_query.lower() else "") + mentioned_day,
                        today_dt,
                    )
                    if corrected:
                        print(f"  [DATE FIX] LLM said {date_str} ({resolved_wd}) but customer said {mentioned_day} → corrected to {corrected}")
                        date_str = corrected
            except (ValueError, TypeError):
                pass

        # Also try pure Python first if query contains day names
        if mentioned_day and not date_str:
            prefix = "next " if "next" in full_query.lower() else ""
            date_str = _resolve_date_python(prefix + mentioned_day, today_dt) or date_str

        try:
            req_dt  = datetime.fromisoformat(f"{date_str}T{time_str}:00")
            day_key = req_dt.strftime("%A")[:3].lower()
        except ValueError:
            return {"success": False, "message": "That date or time didn't look right. Could you try again? (e.g. 'Monday March 23 at 2pm')"}

        # Find / validate provider
        prov_id = booking.get("provider_id")
        assigned_provider = None
        if prov_id:
            p = conn.execute("SELECT * FROM service_providers WHERE id = ?", (prov_id,)).fetchone()
            if p:
                sched = json.loads(p["schedule"]) if p["schedule"] else {}
                if day_key not in sched:
                    return {"success": False,
                            "message": f"Sorry, {p['name']} isn't available on {req_dt.strftime('%A')}s. "
                                       f"They're available: {', '.join(sched.keys())}."}
                assigned_provider = p
        else:
            for p in providers:
                sched = json.loads(p["schedule"]) if p["schedule"] else {}
                if day_key in sched:
                    assigned_provider = p
                    prov_id = p["id"]
                    break
            if not assigned_provider:
                return {"success": False, "message": f"We don't have any staff available on {req_dt.strftime('%A')}s. Would another day work for you?"}

        # Conflict check
        first_svc = conn.execute(
            "SELECT duration_min FROM services WHERE business_name = ? ORDER BY id LIMIT 1",
            (state.get("business_name"),)
        ).fetchone()
        probe_dur = (first_svc["duration_min"] or 30) if first_svc else 30
        new_end_dt = req_dt + timedelta(minutes=probe_dur)

        conflict_row = conn.execute(
            """SELECT o.scheduled_at, COALESCE(s.duration_min, 30) as dur_min
               FROM orders o LEFT JOIN services s ON o.service_id = s.id
               WHERE o.provider_id = ? AND o.status IN ('pending','confirmed')
                 AND datetime(o.scheduled_at) < datetime(?)
                 AND datetime(o.scheduled_at, '+' || COALESCE(s.duration_min,30) || ' minutes') > datetime(?)
               LIMIT 1""",
            (prov_id, new_end_dt.isoformat(), req_dt.isoformat()),
        ).fetchone()

        if conflict_row:
            alts = _find_next_available_slots(conn, prov_id, req_dt, probe_dur, count=3)
            alt_str = ", ".join(a.strftime("%a %b %d at %I:%M %p") for a in alts)
            return {
                "success": False,
                "conflict": True,
                "message": (
                    f"That slot is unfortunately already taken. "
                    f"Here are the next available times with {assigned_provider['name']}: {alt_str or 'please call us'}. "
                    "Would any of those work for you?"
                ),
                "alternatives": [a.isoformat() for a in alts],
            }

        # Auto-select service if none specified
        services_to_book = booking.get("services", [])
        if not services_to_book:
            default_svc = conn.execute(
                "SELECT id, name FROM services WHERE business_name = ? ORDER BY id LIMIT 1",
                (state.get("business_name"),)
            ).fetchone()
            if default_svc:
                services_to_book = [{"service_id": default_svc["id"], "service_name": default_svc["name"]}]
            else:
                return {"success": False, "message": "What service would you like to book?"}

        # Resolve service details
        svc_info = services_to_book[0]
        svc_row = conn.execute(
            "SELECT * FROM services WHERE id = ? AND business_name = ?",
            (svc_info["service_id"], state.get("business_name"))
        ).fetchone()
        if not svc_row:
            return {"success": False, "needs_info": True, "message": ("I could not find that service. Please type the exact service name, e.g. **Dental Cleaning** or **Root Canal Treatment**.")}

        svc_name = svc_row["name"]
        price    = svc_row["price"]
        duration = svc_row["duration_min"] or 30

        # ── SAVE PENDING — do NOT write to orders yet ────────────────────────
        pending_booking = {
            "service_id":    svc_info["service_id"],
            "service_name":  svc_name,
            "price":         price,
            "duration":      duration,
            "provider_id":   prov_id,
            "provider_name": assigned_provider["name"],
            "scheduled_dt":  req_dt.isoformat(),
        }
        conv_state["pending_booking"] = pending_booking
        save_conversation_state(conversation_id, conv_state)

        dt_friendly = req_dt.strftime("%A, %B %d, %Y at %I:%M %p").lstrip("0")
        end_friendly = (req_dt + timedelta(minutes=duration)).strftime("%I:%M %p").lstrip("0")
        return {
            "success": False,          # Not committed yet
            "needs_confirmation": True,
            "pending_booking": pending_booking,
            "message": (
                f"Great news — that slot is available! Here's what I have:\n\n"
                f"  Service:   {svc_name}\n"
                f"  Date/Time: {dt_friendly} – {end_friendly}\n"
                f"  Provider:  {assigned_provider['name']}\n"
                f"  Duration:  {duration} min\n"
                f"  Price:     ${price:.2f}\n\n"
                f"Shall I go ahead and confirm this booking for you?"
            ),
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    try:
        services = conn.execute("SELECT * FROM services WHERE business_name = ? ORDER BY id", (state.get("business_name"),)).fetchall()
        providers = conn.execute(
            "SELECT * FROM service_providers WHERE available = 1 AND business_name = ? ORDER BY rating DESC",
            (state.get("business_name"),)
        ).fetchall()

        if not services:
            return {"success": False, "message": "No services available for booking."}

        # Gather recent conversation context
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in (history or [])[-6:]
        )

        svc_list = "\n".join(f"ID:{s['id']}|{s['name']}|${s['price']}|{s['duration_min']}min" for s in services)
        prov_list = "\n".join(f"ID:{p['id']}|{p['name']}|{p['specialty']}" for p in providers) if providers else "No providers listed"

        parse_prompt = f"""Extract booking details from this conversation:

Conversation:
{history_text}
CUSTOMER: {query}

Available services:
{svc_list}

Available providers:
{prov_list}

Today's date is: {datetime.now().strftime('%Y-%m-%d %A')}

Extract ALL available info. Return ONLY JSON:
- services: list of objects with {{"service_id": id, "service_name": name}}
- date: YYYY-MM-DD (only if customer EXPLICITLY stated a date: today, tomorrow, Monday, March 17, etc.)
- time: HH:MM (only if customer EXPLICITLY stated a time: 2pm, 3:30, morning, etc.)
- provider_id: integer (if specified)
- parsed_ok: boolean
- missing_fields: list of strings (e.g. ["date", "time", "service"])

CRITICAL: Do NOT guess or infer date or time. If the customer did NOT explicitly state a date, include "date" in missing_fields and leave date empty. If the customer did NOT explicitly state a time, include "time" in missing_fields and leave time empty. Set parsed_ok to false if date or time is missing.
If the user only said which service they want (e.g. "I want Men's haircut") without any date or time, return missing_fields: ["date", "time"] and parsed_ok: false.
Interpret relative dates ONLY when the customer explicitly said them (e.g. "tomorrow at 2pm").
No markdown, just JSON."""

        raw = llm_call(parse_prompt, temperature=0.1, max_tokens=500)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            booking = json.loads(raw)
        except json.JSONDecodeError:
            return {"success": False, "needs_info": True, "message": ("Please tell me the service, date and time. Example: **Teeth Cleaning on Monday at 2pm**")}

        missing = booking.get("missing_fields", [])
        if not booking.get("parsed_ok", False) or missing:
            return {
                "success": False,
                "needs_info": True,
                "missing_fields": missing,
                "partial_booking": booking,
                "message": f"I need a bit more information to complete the booking. Please provide: {', '.join(missing)}.",
            }

        date_str = booking.get("date", "")
        time_str = booking.get("time", "")
        if not date_str or not time_str:
             return {"success": False, "needs_info": True, "message": ("Please reply with the date and time. Example: **Monday at 2pm** or **March 25 at 10am**")}
             
        try:
            time_str = time_str[:5]
            req_dt = datetime.fromisoformat(f"{date_str}T{time_str}:00")
            day_key = req_dt.strftime("%A")[:3].lower()
            scheduled_dt = req_dt.isoformat()
        except ValueError:
             return {"success": False, "message": "The date or time format is invalid."}

        booking_results = []
        services_to_book = booking.get("services", [])
        if not services_to_book and booking.get("service_id"):
            services_to_book = [{"service_id": booking["service_id"], "service_name": booking.get("service_name", "Service")}]

        prov_id = booking.get("provider_id")
        assigned_provider = None

        # ── 1. Validate / Find Provider ─────────────────────────────────────
        if prov_id:
            p = conn.execute("SELECT * FROM service_providers WHERE id = ?", (prov_id,)).fetchone()
            if p:
                sched = json.loads(p["schedule"]) if p["schedule"] else {}
                if day_key not in sched:
                    return {
                        "success": False,
                        "message": f"I'm sorry, {p['name']} is not available on {req_dt.strftime('%A')}s. They work: {', '.join(sched.keys())}."
                    }
                assigned_provider = p
        else:
            for p in providers:
                sched = json.loads(p["schedule"]) if p["schedule"] else {}
                if day_key in sched:
                    assigned_provider = p
                    prov_id = p["id"]
                    break
            if not assigned_provider:
                return {"success": False, "message": f"We are closed or have no providers available on {req_dt.strftime('%A')}s."}

        # ── 2. Duration-aware conflict check (runs even if no service specified) ─
        # This is the KEY fix: conflict detection must fire before "no service" return.
        first_svc = conn.execute(
            "SELECT duration_min FROM services WHERE business_name = ? ORDER BY id LIMIT 1",
            (state.get("business_name"),)
        ).fetchone()
        probe_dur = (first_svc["duration_min"] or 30) if first_svc else 30
        new_end_dt = req_dt + timedelta(minutes=probe_dur)

        conflict_row = conn.execute(
            """SELECT o.id, u.full_name as client_name, o.scheduled_at,
                      COALESCE(s.duration_min, 30) as dur_min
               FROM orders o
               LEFT JOIN services s ON o.service_id = s.id
               LEFT JOIN users u ON o.user_id = u.id
               WHERE o.provider_id = ?
                 AND o.status IN ('pending','confirmed')
                 AND datetime(o.scheduled_at) < datetime(?)
                 AND datetime(o.scheduled_at, '+' || COALESCE(s.duration_min, 30) || ' minutes') > datetime(?)
               LIMIT 1""",
            (prov_id, new_end_dt.isoformat(), req_dt.isoformat()),
        ).fetchone()

        if conflict_row:
            pname = assigned_provider["name"] if assigned_provider else "The provider"
            conflict_end = datetime.fromisoformat(conflict_row["scheduled_at"]) + timedelta(minutes=conflict_row["dur_min"])
            alts = _find_next_available_slots(conn, prov_id, req_dt, probe_dur, count=3)
            alt_str = ", ".join(a.strftime("%a %b %d at %I:%M %p") for a in alts)
            return {
                "success": False,
                "conflict": True,
                "message": (
                    f"Sorry, {pname} is not available at {req_dt.strftime('%I:%M %p')} on "
                    f"{req_dt.strftime('%A %B %d')} — there is already a booking from "
                    f"{datetime.fromisoformat(conflict_row['scheduled_at']).strftime('%I:%M %p')} "
                    f"until {conflict_end.strftime('%I:%M %p')}. "
                    f"Next available: {alt_str or 'please call us to check availability'}."
                ),
                "alternatives": [a.isoformat() for a in alts],
            }

        # ── 3. Auto-select default service if none specified ─────────────────
        if not services_to_book:
            default_svc = conn.execute(
                "SELECT id, name FROM services WHERE business_name = ? ORDER BY id LIMIT 1",
                (state.get("business_name"),)
            ).fetchone()
            if default_svc:
                services_to_book = [{"service_id": default_svc["id"], "service_name": default_svc["name"]}]
            else:
                return {"success": False, "message": "No specific services identified for booking. What service would you like?"}

        # 3. Book each service
        user = conn.execute("SELECT full_name, username FROM users WHERE id = ?", (user_id,)).fetchone()
        patient_name = (
            ((user["full_name"] or "").strip() if user else "") or
            (user["username"] if user else "") or
            state.get("customer_context", {}).get("full_name", "") or
            "Customer"
        )
        
        cumulative_msg = ""
        last_ref = ""
        current_dt = req_dt
        audit_logs = []  # Defer logging until after commit to avoid DB lock
        first_order_id = None
        appointment_start = req_dt
        appointment_end = req_dt
        services_summary = []

        for svc_info in services_to_book:
            s_id = svc_info["service_id"]
            svc_row = conn.execute("SELECT * FROM services WHERE id = ? AND business_name = ?", (s_id, state.get("business_name"))).fetchone()
            if not svc_row: continue
            
            svc_name = svc_row["name"]
            price = svc_row["price"]
            duration = svc_row["duration_min"] or 30

            cur = conn.execute(
                """INSERT INTO orders (business_name, user_id, service_id, provider_id, status, order_type, scheduled_at, total_price, notes)
                   VALUES (?, ?, ?, ?, 'confirmed', 'appointment', ?, ?, ?)""",
                (state.get("business_name"), user_id, s_id, prov_id, current_dt.isoformat(), price, f"Booked via AI assistant for {state.get('business_name')}"),
            )
            order_id = cur.lastrowid
            if first_order_id is None:
                first_order_id = order_id
                appointment_start = current_dt
            audit_logs.append(("orders", "INSERT", order_id, {"type": "appointment", "service": svc_name, "date": current_dt.isoformat()}))
            ref = _generate_ref("APPT", order_id)
            last_ref = ref

            end_dt = current_dt + timedelta(minutes=duration)
            appointment_end = end_dt
            services_summary.append(svc_name)
            conn.execute(
                """INSERT INTO calendar_events (business_name, user_id, order_id, title, description, start_time, end_time, provider, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (state.get("business_name"), user_id, order_id, f"Appointment: {svc_name}", f"Booking #{order_id} for {svc_name}. Ref: {ref}", current_dt.isoformat(), end_dt.isoformat(), assigned_provider["name"], 'confirmed'),
            )
            audit_logs.append(("calendar_events", "INSERT", None, {"order_id": order_id, "service": svc_name, "start": current_dt.isoformat()}))
            
            cumulative_msg += f"\n- **{svc_name}**: {current_dt.strftime('%H:%M')} ({duration} min) - ${price:.2f} [Ref: {ref}]"
            # Sequence them back-to-back if multiple
            current_dt = end_dt

        # ONE .ics file per appointment (not per service)
        combined_title = " & ".join(services_summary) if services_summary else "Appointment"
        _generate_ics(
            first_order_id, combined_title, appointment_start, appointment_end,
            assigned_provider["name"], patient_name,
            business_name=state.get("business_name", ""),
            method="REQUEST", sequence=0,
        )

        conn.commit()
        conn.close()
        conn = None  # Ensure finally doesn't double-close
        # Audit logging AFTER commit/close to avoid DB lock contention
        conv_id = state.get("conversation_id", "")
        for tbl, op, rid, kv in audit_logs:
            try:
                log_db_write(conv_id, tbl, op, rid, kv)
            except Exception:
                pass
        # Build ics_file_url (one file per appointment, uses first order_id)
        safe_name = re.sub(r"[^a-z0-9_]", "_", patient_name.lower())
        ics_filename = f"{safe_name}_appointment_{first_order_id}.ics"
        return {
            "success": True,
            "message": f" Appointments confirmed for {patient_name} with {assigned_provider['name']}:{cumulative_msg}",
            "booking_ref": last_ref,
            "booking_id": first_order_id,
            "provider": assigned_provider["name"],
            "date": date_str,
            "ics_file_url": f"/calendar/{ics_filename}",
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def reschedule_booking(state: dict, **kwargs) -> dict:
    """Reschedule an existing booking, parsing the new date/time from conversation."""
    user_id = state.get("user_id")
    query = state.get("query", "")
    history = state.get("history", [])
    conn = _db()
    try:
        order = conn.execute(
            """SELECT o.*, s.name as service_name
               FROM orders o LEFT JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? AND o.status IN ('pending','confirmed')
               AND s.business_name = ?
               ORDER BY o.scheduled_at DESC LIMIT 1""",
            (user_id, state.get("business_name")),
        ).fetchone()
        if not order:
            return {"success": False, "message": "No active bookings found to reschedule."}

        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in (history or [])[-4:])

        parse_prompt = (
            "Extract the NEW date and time the customer wants to reschedule to.\n"
            "\n"
            f"Conversation:\n{history_text}\n"
            f"CUSTOMER: {query}\n"
            "\n"
            f"Current booking: {order['service_name']} on {order['scheduled_at']}\n"
            f"Today is: {datetime.now().strftime('%Y-%m-%d %A')}\n"
            "\n"
            "RULES:\n"
            "- Only set date if the customer EXPLICITLY stated a new date (today, tomorrow, Monday, March 5, etc.).\n"
            "- Only set time if the customer EXPLICITLY stated a new time (3pm, 10:30, morning, etc.).\n"
            "- Do NOT guess, infer, or assume a date or time.\n"
            "- Interpret relative dates correctly using today's date above.\n"
            "- If date or time is missing, set parsed_ok to false.\n"
            "\n"
            'Return ONLY JSON: {"date": "2025-03-05", "time": "15:00", "parsed_ok": true}\n'
            'If not enough info: {"parsed_ok": false, "message": "When would you like to reschedule to?"}\n'
            "No markdown. No prose. Just JSON."
        )

        raw = llm_call(parse_prompt, temperature=0.1, max_tokens=150)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "success": False,
                "needs_info": True,
                "message": (
                    "Please reply with your new preferred date and time. "
                    "For example: **Wednesday at 3pm** or **March 25 at 10am**"
                ),
            }

        if not result.get("parsed_ok", False):
            return {
                "success": False,
                "needs_info": True,
                "message": (
                    "Please tell me the new date and time for your appointment. "
                    "For example: **Monday at 2pm** or **March 25 at 10:00**"
                ),
            }

        new_date = result.get("date", "")
        new_time = result.get("time", "")
        new_dt = f"{new_date}T{new_time}:00"

        conn.execute(
            "UPDATE orders SET scheduled_at = ?, notes = ? WHERE id = ?",
            (new_dt, "Rescheduled via AI assistant", order["id"]),
        )

        # Get actual service duration from DB
        svc_dur = conn.execute(
            "SELECT duration_min FROM services WHERE id = ?", (order["service_id"],)
        ).fetchone()
        dur_min = (svc_dur["duration_min"] or 30) if svc_dur else 30

        # Get provider name
        prov_row = conn.execute(
            "SELECT name FROM service_providers WHERE id = ?", (order["provider_id"],)
        ).fetchone()
        provider_name = prov_row["name"] if prov_row else "Assigned Staff"

        # Get patient name (full_name, else username, else customer_context, else "Customer")
        user_row = conn.execute("SELECT full_name, username FROM users WHERE id = ?", (user_id,)).fetchone()
        patient_name = (
            ((user_row["full_name"] or "").strip() if user_row else "") or
            (user_row["username"] if user_row else "") or
            state.get("customer_context", {}).get("full_name", "") or
            "Customer"
        )

        # Update calendar event
        try:
            start = datetime.fromisoformat(new_dt)
            end = start + timedelta(minutes=dur_min)
            conn.execute(
                "UPDATE calendar_events SET start_time = ?, end_time = ?, status = 'confirmed' WHERE order_id = ?",
                (start.isoformat(), end.isoformat(), order["id"]),
            )
        except ValueError:
            pass

        conn.commit()
        conn.close()
        conn = None
        # Audit logging AFTER commit/close to avoid DB lock
        try:
            log_db_write(state.get("conversation_id", ""), "orders", "UPDATE", order["id"], {"action": "reschedule", "new_dt": new_dt})
            log_db_write(state.get("conversation_id", ""), "calendar_events", "UPDATE", None, {"order_id": order["id"], "action": "reschedule", "new_start": new_dt})
        except Exception:
            pass

        # ── ICS: cancel old event then write updated event ───────────────────
        try:
            start = datetime.fromisoformat(new_dt)
            end = start + timedelta(minutes=dur_min)
            biz_name = state.get("business_name", "")

            # Step 1: CANCEL the old event (SEQUENCE=1 so calendar app removes it)
            _generate_ics(
                order["id"], order["service_name"], start, end,
                provider_name, patient_name,
                business_name=biz_name,
                method="CANCEL", sequence=1,
            )

            # Step 2: REQUEST the new event (SEQUENCE=2 so calendar app adds new time)
            _generate_ics(
                order["id"], order["service_name"], start, end,
                provider_name, patient_name,
                business_name=biz_name,
                method="REQUEST", sequence=2,
            )
        except Exception:
            pass

        booking_ref = _generate_ref("APPT", order["id"])

        return {
            "success": True,
            "booking_id": order["id"],
            "booking_ref": booking_ref,
            "service": order["service_name"],
            "new_date": new_date,
            "new_time": new_time,
            "message": f" Appointment {booking_ref} for {order['service_name']} has been rescheduled to {new_date} at {new_time}.",
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def cancel_booking(state: dict, **kwargs) -> dict:
    """
    Cancel an existing booking.  Two-phase:
      Phase 1 — find active booking, show details, ask for confirmation.
      Phase 2 — user confirmed → mark cancelled in DB.
    """
    user_id         = state.get("user_id")
    query           = state.get("query", "").lower()
    conversation_id = state.get("conversation_id", "")

    from core.database import load_conversation_state, save_conversation_state
    conv_state = load_conversation_state(conversation_id) or {}
    pending_cancel = conv_state.get("pending_cancel")

    is_confirmed = pending_cancel and any(w in query for w in [
        "yes", "confirm", "cancel it", "go ahead", "sure", "ok", "okay",
        "correct", "please", "do it", "yes cancel", "confirm cancel",
    ])

    conn = _db()
    try:
        # ── Phase 2: commit cancellation ────────────────────────────────────
        if is_confirmed and pending_cancel:
            order_ids    = pending_cancel["order_ids"]
            services_str = pending_cancel["services_str"]
            dt_str       = pending_cancel["dt_str"]
            patient_name = pending_cancel["patient_name"]
            provider_name = pending_cancel["provider_name"]
            main_order_id = pending_cancel["main_order_id"]

            # Re-fetch to make sure they're still active
            still_active = conn.execute(
                "SELECT id FROM orders WHERE id IN ({}) AND status IN ('pending','confirmed')".format(
                    ",".join("?" * len(order_ids))
                ),
                order_ids
            ).fetchall()

            if not still_active:
                conv_state.pop("pending_cancel", None)
                save_conversation_state(conversation_id, conv_state)
                return {"success": False,
                        "message": "It looks like that appointment was already cancelled or doesn't exist."}

            for oid in order_ids:
                conn.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (oid,))
                conn.execute("UPDATE calendar_events SET status = 'cancelled' WHERE order_id = ?", (oid,))
            conn.commit()
            conn.close()
            conn = None

            # Delete .ics files and generate cancel ICS
            safe_name    = re.sub(r"[^a-z0-9_]", "_", patient_name.lower())
            _proj_root   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            calendar_dir = os.path.join(_proj_root, "calendar")
            for oid in order_ids:
                p = os.path.join(calendar_dir, f"{safe_name}_appointment_{oid}.ics")
                if os.path.exists(p):
                    try: os.remove(p)
                    except OSError: pass

            cancel_path = None
            try:
                first_start = datetime.fromisoformat(pending_cancel["scheduled_dt"])
                last_end    = first_start + timedelta(minutes=pending_cancel["duration"])
                cancel_path = _generate_ics(
                    main_order_id, services_str, first_start, last_end,
                    provider_name, patient_name,
                    business_name=state.get("business_name", ""),
                    method="CANCEL", sequence=1,
                )
            except Exception:
                pass

            booking_ref = _generate_ref("APPT", main_order_id)
            conv_state.pop("pending_cancel", None)
            save_conversation_state(conversation_id, conv_state)

            try:
                for oid in order_ids:
                    log_db_write(conversation_id, "orders", "UPDATE", oid,
                                 {"action": "cancel", "status": "cancelled"})
            except Exception:
                pass

            res = {
                "success": True,
                "cancelled_booking_id": main_order_id,
                "booking_ref": booking_ref,
                "service": services_str,
                "scheduled_at": dt_str,
                "message": (
                    f"Done — your appointment has been cancelled.\n"
                    f"  Service:   {services_str}\n"
                    f"  Was:       {dt_str}\n"
                    f"  Reference: {booking_ref}\n"
                    "The slot is now free. If you'd like to rebook at a different time, just let me know!"
                ),
            }
            if cancel_path:
                res["cancel_ics_url"] = f"/calendar/{os.path.basename(cancel_path)}"
            return res

        # ── Phase 1: find booking and ask for confirmation ───────────────────
        order = conn.execute(
            """SELECT o.*, s.name as service_name, s.duration_min
               FROM orders o LEFT JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? AND o.status IN ('pending', 'confirmed')
               AND s.business_name = ?
               ORDER BY o.scheduled_at DESC LIMIT 1""",
            (user_id, state.get("business_name")),
        ).fetchone()

        if not order:
            return {"success": False, "message": ("I could not find any active bookings to cancel. If you want to make a new booking, type the service name and date, e.g. **Dental Cleaning on Monday at 2pm**.")}

        order_date     = order["scheduled_at"][:10] if order["scheduled_at"] else ""
        session_orders = conn.execute(
            """SELECT o.id, o.scheduled_at, s.name as service_name, s.duration_min
               FROM orders o LEFT JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? AND o.provider_id = ? AND o.status IN ('pending','confirmed')
               AND o.scheduled_at LIKE ? AND s.business_name = ?
               ORDER BY o.scheduled_at ASC""",
            (user_id, order["provider_id"], f"{order_date}%", state.get("business_name")),
        ).fetchall()

        order_ids    = [r["id"] for r in session_orders]
        services_str = ", ".join(r["service_name"] for r in session_orders)

        prov_row = conn.execute("SELECT name FROM service_providers WHERE id = ?",
                                (order["provider_id"],)).fetchone()
        provider_name = prov_row["name"] if prov_row else "your provider"

        user_row = conn.execute("SELECT full_name, username FROM users WHERE id = ?",
                                (user_id,)).fetchone()
        patient_name = (
            ((user_row["full_name"] or "").strip() if user_row else "") or
            (user_row["username"] if user_row else "") or "Customer"
        )

        try:
            dt = datetime.fromisoformat(order["scheduled_at"])
            dt_str = dt.strftime("%A, %B %d at %I:%M %p").lstrip("0")
        except Exception:
            dt_str = order["scheduled_at"]

        duration = order["duration_min"] or 30

        # Save pending cancel state (don't modify DB yet)
        conv_state["pending_cancel"] = {
            "order_ids":    order_ids,
            "main_order_id": order["id"],
            "services_str": services_str,
            "dt_str":       dt_str,
            "patient_name": patient_name,
            "provider_name": provider_name,
            "scheduled_dt": order["scheduled_at"],
            "duration":     duration,
        }
        save_conversation_state(conversation_id, conv_state)

        booking_ref = _generate_ref("APPT", order["id"])
        return {
            "success": False,
            "needs_confirmation": True,
            "pending_cancel": conv_state["pending_cancel"],
            "message": (
                f"I found your appointment. Here are the details:\n\n"
                f"  Service:   {services_str}\n"
                f"  Date/Time: {dt_str}\n"
                f"  Provider:  {provider_name}\n"
                f"  Reference: {booking_ref}\n\n"
                "Are you sure you'd like to cancel this? Just say **yes** to confirm, or **no** to keep it."
            ),
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

def check_availability(state: dict, **kwargs) -> dict:
    """Check available time slots, considering existing bookings and provider schedules."""
    query = state.get("query", "")
    conn = _db()
    try:
        providers = conn.execute(
            "SELECT * FROM service_providers WHERE available = 1 AND business_name = ? ORDER BY rating DESC",
            (state.get("business_name"),)
        ).fetchall()
        services = conn.execute(
            "SELECT id, name, duration_min, price FROM services WHERE business_name = ?",
            (state.get("business_name"),)
        ).fetchall()

        # Get existing bookings for next 7 days (use local time to match seed data)
        business_name = state.get("business_name", "")
        now = datetime.now()  # local time, matches synthetic data seeding
        booked_slots = []
        for day_offset in range(1, 8):
            date = (now + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            existing = conn.execute(
                """SELECT scheduled_at, provider_id FROM orders
                   WHERE business_name = ? AND status IN ('pending','confirmed') AND scheduled_at LIKE ?""",
                (business_name, f"{date}%"),
            ).fetchall()
            booked_slots.extend([(e["scheduled_at"], e["provider_id"]) for e in existing])

        # Generate available slots
        slots = []
        for day_offset in range(1, 8):
            dt = now + timedelta(days=day_offset)
            date_str = dt.strftime("%Y-%m-%d")
            day_name = dt.strftime("%A")
            day_key = day_name[:3].lower() # mon, tue, wed...
            
            # Find providers available on THIS day
            available_today = []
            for p in providers:
                sched = json.loads(p["schedule"]) if p["schedule"] else {}
                if day_key in sched:
                    available_today.append(p)
            
            if not available_today:
                continue

            # Parse provider hour ranges for this day (e.g. "9-17" -> 9 to 16 inclusive)
            hour_min, hour_max = 9, 17
            for p in available_today:
                sched = json.loads(p["schedule"]) if p["schedule"] else {}
                rng = sched.get(day_key, "")
                if rng and "-" in rng:
                    try:
                        a, b = rng.replace(" ", "").split("-")
                        hour_min = min(hour_min, int(a))
                        hour_max = max(hour_max, int(b))
                    except (ValueError, TypeError):
                        pass

            for hour in range(hour_min, hour_max):
                slot_str = f"{date_str} {hour:02d}:00"
                # Check if slot is not fully booked across all available providers
                bookings_at_slot = sum(1 for s, pid in booked_slots if s and slot_str.replace(" ", "T") in s and any(ap["id"] == pid for ap in available_today))
                if bookings_at_slot < len(available_today):
                    slots.append({
                        "datetime": slot_str, 
                        "day": day_name, 
                        "available_providers_count": len(available_today) - bookings_at_slot,
                        "providers": [p["name"] for p in available_today]
                    })

        # Build per-day summary for the message so the LLM gives accurate numbers
        day_counts: dict[str, int] = {}
        for slot in slots:
            day = slot["day"]
            day_counts[day] = day_counts.get(day, 0) + 1

        day_summary = ", ".join(
            f"{day}: {cnt} slot{'s' if cnt != 1 else ''}"
            for day, cnt in day_counts.items()
        )

        return {
            "success": True,
            "available_providers": [{"name": p["name"], "specialty": p["specialty"], "rating": p["rating"]} for p in providers],
            "available_slots": slots[:20],
            "services": [dict(s) for s in services],
            "slots_by_day": day_counts,
            "message": (
                f"{len(providers)} providers available. "
                f"{len(slots)} open slots across the next 7 days "
                f"({day_summary})."
            ),
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Address Validation Tool
# ─────────────────────────────────────────────

def validate_address(state: dict, **kwargs) -> dict:
    """Extract, validate, and normalise delivery address from conversation."""
    query = state.get("query", "")
    history = state.get("history", [])

    history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in (history or [])[-6:])

    validate_prompt = f"""Extract and validate the delivery address from this conversation:

Conversation:
{history_text}
CUSTOMER: {query}

Extract the following. Return ONLY JSON:
{{
  "found": true,
  "street": "34 Front Street",
  "unit": "",
  "city": "Toronto",
  "province": "ON",
  "postal_code": "M4K 6B2",
  "country": "Canada",
  "formatted": "34 Front Street, Toronto, ON M4K 6B2, Canada",
  "postal_code_valid": true,
  "issues": []
}}

Rules for Canadian postal codes: format A1A 1A1 (letter-digit-letter space digit-letter-digit).
CROSS-REFERENCE CITY/PROVINCE: Ensure the first letter of the postal code matches the province (e.g. M=Toronto, K/N=Ontario, T=Alberta).
If the address is valid but the postal code is slightly formatted differently (e.g. "M4K6B2" vs "M4K 6B2"), normalize it to "A1A 1A1" and set valid=true.
If no address found: {{"found": false, "message": "Please provide your delivery address."}}
No markdown, just JSON."""

    raw = llm_call(validate_prompt, temperature=0.1, max_tokens=300)
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"success": False, "needs_info": True, "message": ("Please type your full delivery address. Example: **123 Main Street, Toronto, ON M5V 1A1**")}

    if not result.get("found", False):
        return {"success": False, "needs_info": True, "message": result.get("message") or "Please type your full delivery address. Example: **123 Main Street, Toronto, ON M5V 1A1**"}

    address_str = result.get("formatted", "")
    postal_code = result.get("postal_code", "").upper().replace(" ", "")
    issues = result.get("issues", [])

    # Robust Regex Validation for Canadian Postal Codes
    is_canadian = result.get("country", "").lower() == "canada" or result.get("province", "") in ["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "PE", "NL", "YT", "NT", "NU"]
    if is_canadian and postal_code:
        if not re.match(r"^[A-Z]\d[A-Z]\d[A-Z]\d$", postal_code):
            issues.append("Invalid Canadian postal code format. Should be A1A 1A1.")
        else:
            # Re-format with space
            result["postal_code"] = f"{postal_code[:3]} {postal_code[3:]}"

    # Real existence check with Geopy
    if GEOPY_AVAILABLE and address_str and not issues:
        try:
            geolocator = Nominatim(user_agent="generic_ai_agent_business")
            location = geolocator.geocode(address_str, timeout=5)
            if not location:
                # Try without unit if present
                simplified = f"{result.get('street', '')}, {result.get('city', '')}, {result.get('province', '')}, Canada"
                location = geolocator.geocode(simplified, timeout=5)
            
            if not location:
                # If it's a very specific address, Nominatim sometimes fails. 
                # If the postal code is valid and city matches, we can be more lenient.
                pass 
            else:
                # Basic postal code confront (first 3 chars)
                pc_prefix = result.get("postal_code", "")[:3].upper()
                if pc_prefix and pc_prefix not in location.address.upper():
                    # Some Nominatim responses are weird, double check
                    if result.get("postal_code", "").replace(" ", "").upper() not in location.address.replace(" ", "").upper():
                         # Only flag if it's a major mismatch
                         if result.get("city", "").lower() not in location.address.lower():
                            issues.append(f"Postal code {result['postal_code']} might not match this location ({location.address[:50]}...)")
        except (GeocoderTimedOut, GeocoderServiceError):
            pass # Fallback to LLM validation if service is down

    if issues:
        return {
            "success": False,
            "address": result,
            "issues": issues,
            "message": f"I had a bit of trouble verifying that address: {'; '.join(issues)}. Could you please double-check the spelling or postal code?",
        }

    # Store address on the user profile
    user_id = state.get("user_id")
    if user_id:
        conn = _db()
        try:
            conn.execute(
                "UPDATE users SET address = ?, postal_code = ?, city = ? WHERE id = ?",
                (result.get("formatted", ""), result.get("postal_code", ""), result.get("city", ""), user_id),
            )
            log_db_write(state.get("conversation_id", ""), "users", "UPDATE", user_id, {"action": "address_update", "address": result.get("formatted", "")})
            conn.commit()
        finally:
            conn.close()

    return {
        "success": True,
        "address": result,
        "formatted_address": result.get("formatted", ""),
        "postal_code_valid": result.get("postal_code_valid", True),
        "message": f"Address confirmed: {result.get('formatted', '')}",
    }


# ─────────────────────────────────────────────
# Delivery Type Tool
# ─────────────────────────────────────────────

def set_delivery_type(state: dict, **kwargs) -> dict:
    """Set whether the order is for pickup or delivery."""
    # Priority 1: explicit argument from LLM function call
    delivery_type_arg = (
        state.get("delivery_type") or kwargs.get("delivery_type") or ""
    ).lower().strip()

    # Priority 2: scan current query
    query = state.get("query", "").lower()

    # Priority 3: scan recent history for delivery intent
    history = state.get("history", [])
    history_text = " ".join(
        m.get("content", "") for m in history[-6:] if m.get("role") == "user"
    ).lower()

    combined = f"{delivery_type_arg} {query} {history_text}"

    if any(w in combined for w in ["pickup", "pick up", "pick-up", "collect", "come get", "i'll pick", "pick it"]):
        delivery_type = "pickup"
    elif any(w in combined for w in ["deliver", "delivery", "bring", "send", "home", "address"]):
        delivery_type = "delivery"
    else:
        return {
            "success": False,
            "needs_info": True,
            "message": (
                "How would you like to receive your order? "
                "Type **pickup** to collect it yourself, "
                "or **delivery** to have it sent to your address."
            ),
        }

    if delivery_type == "delivery":
        return {
            "success": True,
            "delivery_type": "delivery",
            "needs_address": True,
            "message": "Got it — delivery! Please provide your full delivery address.",
        }
    else:
        return {
            "success": True,
            "delivery_type": "pickup",
            "needs_address": False,
            "message": "Got it — pickup! Your order will be ready for collection.",
        }

def set_delivery_address(state: dict, **kwargs) -> dict:
    """Set delivery address WITH validation."""
    from processing.address_validator import extract_and_validate_address
    
    query = state.get("query", "")
    user_id = state.get("user_id")
    
    # Parse and validate
    address_data = extract_and_validate_address(query)
    
    if not address_data or not address_data.get("valid"):
        msg = address_data.get("message") if address_data else "Could not parse address"
        return {
            "success": False,
            "message": msg
        }
    
    # Save to database
    conn = _db()
    try:
        conn.execute(
            "UPDATE users SET address = ?, postal_code = ?, city = ? WHERE id = ?",
            (address_data["address"], address_data["postal_code"], 
             address_data.get("city"), user_id)
        )
        conn.commit()
        
        return {
            "success": True,
            "address": address_data["address"],
            "postal_code": address_data["postal_code"],
            "city": address_data["city"],
            "message": f"✓ Address confirmed:\n{address_data['message']}"
        }
    except Exception as e:
        return {"success": False, "message": f"Error saving address: {e}"}
    finally:
        conn.close()

# ─────────────────────────────────────────────
# Information & History Tools
# ─────────────────────────────────────────────

def _extract_active_filter(state: dict) -> str:
    """
    Extract and persist an active service/category filter from the conversation.

    Logic:
    1. Check conversation_state for a saved filter ("active_service_filter").
    2. If the current query narrows or sets a new filter, update conversation_state.
    3. If the current query is a vague follow-up ("cheapest", "which is best",
       "show me all") return the SAVED filter so context is preserved.
    4. If the query explicitly widens scope ("show everything", "all services"),
       clear the saved filter.

    Returns the active filter string (may be empty).
    """
    from core.database import load_conversation_state, save_conversation_state

    conv_id = state.get("conversation_id", "")
    query   = state.get("query", "").lower().strip()

    cs      = load_conversation_state(conv_id) or {}
    saved   = cs.get("active_service_filter", "")

    # Vague follow-up words — preserve existing filter, never re-derive
    VAGUE_WORDS = {
        # price/value queries
        "cheap", "cheapest", "cheapest one", "cheap one", "most expensive",
        "affordable", "budget", "price", "cost", "how much", "lowest",
        "tell me cheap", "tell me the cheapest", "what's the cheapest",
        "which is cheapest", "which is best", "which is cheapest?",
        # generic follow-ups
        "best", "popular", "top", "recommended",
        "recommend", "suggest", "which", "options",
        "show me", "list", "give me", "all of them",
        "what are", "tell me", "more", "others", "other",
    }
    # Also treat any query under 5 words that has a saved filter as vague
    is_short_query = len(query.split()) <= 4
    if saved and (
        any(query == w or query.startswith(w + " ") or query == w + "?" for w in VAGUE_WORDS)
        or is_short_query
    ):
        return saved

    # Explicit reset words — clear filter
    RESET = {"all services", "everything", "full menu", "full list", "show everything"}
    if any(r in query for r in RESET):
        cs.pop("active_service_filter", None)
        save_conversation_state(conv_id, cs)
        return ""

    # Try to extract a new filter from the query using LLM
    history = state.get("history", [])
    recent_user_msgs = " | ".join(
        m.get("content", "")
        for m in history[-6:] if m.get("role") == "user"
    )
    all_chunks = state.get("all_chunks", [])
    services_sample = ""
    try:
        conn = _db()
        svcs = conn.execute(
            "SELECT name, category FROM services WHERE business_name = ? ORDER BY id LIMIT 30",
            (state.get("business_name"),),
        ).fetchall()
        conn.close()
        services_sample = ", ".join(
            f"{s['name']} ({s['category'] or 'general'})" for s in svcs
        )
    except Exception:
        pass

    filter_prompt = f"""You are analysing what category or topic a customer is focused on.

Recent conversation (user messages):
{recent_user_msgs}

Current query: "{query}"

Available services/categories: {services_sample or '(unknown)'}

Identify the service topic/category filter implied by the conversation.
Examples:
- "I need something for my feet" → "feet"
- "tell me about massages" → "massage"
- "any hair treatments?" → "hair"
- "show me dental cleaning options" → "dental cleaning"
- "what do you have?" → ""   (no filter — too vague)
- "cheapest" → ""            (vague follow-up — no new filter)

Return ONLY the filter keyword(s) as a short string, or "" if none.
No quotes, no JSON, just the filter text or blank."""

    raw = llm_call(filter_prompt, temperature=0.1, max_tokens=30).strip().strip('"').strip("'")
    new_filter = raw if raw and raw not in ("none", "null", "n/a", '""', "''") else ""

    if new_filter:
        cs["active_service_filter"] = new_filter
        save_conversation_state(conv_id, cs)
        return new_filter

    return saved


def _filter_services(services: list[dict], active_filter: str) -> list[dict]:
    """Return services that match the active filter (name or category contains it)."""
    if not active_filter:
        return services
    kw = active_filter.lower()
    matched = [
        s for s in services
        if kw in (s.get("name") or "").lower()
        or kw in (s.get("category") or "").lower()
        or kw in (s.get("description") or "").lower()
    ]
    return matched if matched else services   # fall back to all if nothing matches


def get_pricing(state: dict, **kwargs) -> dict:
    """Get pricing information for services — respects active category filter from memory."""
    conn = _db()
    try:
        services = conn.execute(
            "SELECT name, description, price, duration_min, category, modifiers FROM services WHERE business_name = ? ORDER BY category, price",
            (state.get("business_name"),)
        ).fetchall()
        services_list = [dict(s) for s in services]

        # ── Apply context filter (memory fix) ──────────────────────────────
        active_filter = _extract_active_filter(state)
        if active_filter:
            filtered = _filter_services(services_list, active_filter)
            filter_note = f"(filtered to: '{active_filter}')"
        else:
            filtered = services_list
            filter_note = ""

        categorised: dict[str, list] = {}
        for svc in filtered:
            cat = svc.get("category") or "General"
            categorised.setdefault(cat, []).append(svc)

        return {
            "success":              True,
            "services_by_category": categorised,
            "total_services":       len(filtered),
            "active_filter":        active_filter,
            "message": (
                f"Found {len(filtered)} service(s) {filter_note}."
                if active_filter else
                f"Found {len(filtered)} services across {len(categorised)} categories."
            ),
        }
    finally:
        conn.close()



def get_recommendations(state: dict, **kwargs) -> dict:
    """Provide personalised recommendations — respects active category filter from memory."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        # ── Apply context filter (memory fix) ──────────────────────────────
        active_filter = _extract_active_filter(state)

        base_where = "s.business_name = ?"
        base_args  = [state.get("business_name")]

        # Build SQL fragment that filters by category/name if a filter is active
        if active_filter:
            kw = f"%{active_filter}%"
            filter_sql  = " AND (LOWER(s.name) LIKE ? OR LOWER(s.category) LIKE ? OR LOWER(s.description) LIKE ?)"
            filter_args = [kw, kw, kw]
        else:
            filter_sql  = ""
            filter_args = []

        # What they've ordered before (filtered)
        past_services = conn.execute(
            f"""SELECT s.name, COUNT(o.id) as cnt FROM orders o
               JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? AND {base_where}{filter_sql}
               GROUP BY s.name ORDER BY cnt DESC LIMIT 5""",
            (user_id, *base_args, *filter_args),
        ).fetchall()

        # Popular items globally for this business (filtered)
        popular = conn.execute(
            f"""SELECT s.name, s.price, COUNT(o.id) as cnt FROM orders o
               JOIN services s ON o.service_id = s.id
               WHERE {base_where}{filter_sql}
               GROUP BY s.name ORDER BY cnt DESC LIMIT 5""",
            (*base_args, *filter_args),
        ).fetchall()

        # New services they haven't tried (filtered)
        used_ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT service_id FROM orders WHERE user_id = ?", (user_id,)
        ).fetchall()]
        if used_ids:
            placeholders = ",".join("?" * len(used_ids))
            new_services = conn.execute(
                f"SELECT * FROM services WHERE {base_where} AND id NOT IN ({placeholders}){filter_sql.replace('s.', '')} ORDER BY price ASC LIMIT 4",
                (*base_args, *used_ids, *filter_args),
            ).fetchall()
        else:
            new_services = conn.execute(
                f"SELECT * FROM services WHERE {base_where}{filter_sql.replace('s.', '')} ORDER BY price ASC LIMIT 4",
                (*base_args, *filter_args),
            ).fetchall()

        loyalty = conn.execute("SELECT points, tier FROM loyalty_points WHERE user_id = ?", (user_id,)).fetchone()

        user = conn.execute("SELECT family_members FROM users WHERE id = ?", (user_id,)).fetchone()
        family_info = []
        if user and user["family_members"]:
            try:
                family_info = json.loads(user["family_members"])
            except json.JSONDecodeError:
                pass

        filter_note = f"(scoped to: '{active_filter}')" if active_filter else ""
        return {
            "success":        True,
            "past_favorites": [{"name": r["name"], "order_count": r["cnt"]} for r in past_services],
            "popular_items":  [{"name": r["name"], "price": r["price"], "order_count": r["cnt"]} for r in popular],
            "new_to_try":     [dict(s) for s in new_services],
            "loyalty_points": loyalty["points"] if loyalty else 0,
            "loyalty_tier":   loyalty["tier"] if loyalty else "bronze",
            "family_members": family_info,
            "active_filter":  active_filter,
            "message": (
                f"Here are personalised recommendations {filter_note}."
                if active_filter else
                "Based on your history, here are some personalised recommendations."
            ),
        }
    finally:
        conn.close()



def get_order_history(state: dict, **kwargs) -> dict:
    """Retrieve the customer's past orders and bookings with full readable details."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        orders = conn.execute(
            """SELECT o.id, o.status, o.order_type, o.scheduled_at, o.total_price,
                      o.delivery_type, o.items_json,
                      s.name as service_name, sp.name as provider_name
               FROM orders o
               LEFT JOIN services s ON o.service_id = s.id
               LEFT JOIN service_providers sp ON o.provider_id = sp.id
               WHERE o.user_id = ? AND s.business_name = ?
               ORDER BY o.scheduled_at DESC LIMIT 10""",
            (user_id, state.get("business_name")),
        ).fetchall()
        order_list = [dict(o) for o in orders]

        if not order_list:
            return {
                "success": True,
                "orders": [],
                "total": 0,
                "message": "You have no previous bookings or orders with us yet. This is your first time — welcome!",
            }

        # Build a human-readable summary so the LLM doesn't have to parse raw dicts
        lines = []
        for o in order_list:
            ref = _generate_ref("APPT" if o.get("order_type") == "appointment" else "ORDER", o["id"])
            svc = o.get("service_name") or "Service"
            provider = o.get("provider_name") or ""
            status = (o.get("status") or "unknown").upper()
            total = f"${o['total_price']:.2f}" if o.get("total_price") else ""
            try:
                dt = datetime.fromisoformat(o["scheduled_at"]).strftime("%b %d, %Y at %I:%M %p") if o.get("scheduled_at") else "N/A"
            except Exception:
                dt = o.get("scheduled_at", "N/A")
            line = f"• {ref} | {svc}"
            if provider:
                line += f" with {provider}"
            line += f" | {dt} | {status}"
            if total:
                line += f" | {total}"
            lines.append(line)

        summary = "\n".join(lines)
        out = {
            "success": True,
            "orders": order_list,
            "total": len(order_list),
            "summary": summary,
            "message": (
                f"Here are your {len(order_list)} previous booking(s) with us:\n{summary}"
            ),
        }
        # Cost breakdown for most recent order (for "cost breakdown" / "my order total" queries)
        try:
            recent = order_list[0]
            ij = recent.get("items_json")
            if ij:
                items = json.loads(ij) if isinstance(ij, str) else ij
                lines_breakdown = []
                for it in items:
                    name = it.get("name", "Item")
                    qty = it.get("qty", 1)
                    price = float(it.get("price", 0))
                    lines_breakdown.append(f"  • {qty}x {name}: ${price * qty:.2f}")
                total_price = recent.get("total_price") or 0
                out["recent_order_cost_breakdown"] = "\n".join(lines_breakdown) + f"\n  Total: ${total_price:.2f}"
        except Exception:
            pass
        return out
    finally:
        conn.close()


def search_knowledge(state: dict, **kwargs) -> dict:
    """Search the business knowledge base (PDF + enriched) for relevant information."""
    from agent.rag_engine import chunk_query_into_topics, retrieve_relevant_chunks
    query = state.get("query", "")
    history = state.get("history", [])
    all_chunks = state.get("all_chunks", [])
    if not all_chunks:
        return {"success": True, "knowledge": "", "message": "No knowledge base loaded."}
    topics = chunk_query_into_topics(query, history)
    chunks, _ = retrieve_relevant_chunks(
        topics, all_chunks, top_n=5, business_name=state.get("business_name", "")
    )
    text = "\n\n".join(f"{c.get('section_title','')}: {c.get('text','')}" for c in chunks)
    return {"success": True, "knowledge": text, "chunks_found": len(chunks), "message": f"Found {len(chunks)} relevant knowledge chunks."}


def get_business_info(state: dict, **kwargs) -> dict:
    """Return combined business information: hours, providers, FAQs, promotions."""
    hours_res = get_business_hours(state, **kwargs)
    prov_res = get_provider_info(state, **kwargs)
    faq_res = get_faqs(state, **kwargs)
    promo_res = get_promotions(state, **kwargs)
    biz_name = state.get("business_name") or get_business_meta("business_name", state.get("business_name","")) or "Business"
    msg = f"**{biz_name}**\n\n"
    msg += f"Hours:\n{hours_res.get('hours','N/A')}\n\n"
    msg += f"Providers: {prov_res.get('total_providers',0)}\n"
    if prov_res.get("providers"):
        msg += "  " + ", ".join(p.get("name","") for p in prov_res["providers"][:5]) + "\n\n"
    msg += f"FAQs: {len(faq_res.get('faqs',[]))} entries\n"
    msg += f"Promotions: {len(promo_res.get('promotions',[]))} active"
    return {"success": True, "business_name": biz_name, "message": msg, "hours": hours_res.get("hours"), "providers": prov_res.get("providers", [])}


def get_business_hours(state: dict, **kwargs) -> dict:
    """Return the business operating hours."""
    biz = state.get("business_name", "")
    hours = get_business_meta("business_hours", biz)
    if not hours:
        hours = "Monday–Friday: 9:00 AM – 7:00 PM\nSaturday: 9:00 AM – 5:00 PM\nSunday: Closed"
    return {"success": True, "hours": hours, "message": f"Business hours:\n{hours}"}


def get_provider_info(state: dict, **kwargs) -> dict:
    """Get information about service providers/staff members. Filters by name if provided."""
    conn = _db()
    try:
        all_providers = conn.execute(
            "SELECT name, specialty, rating, available, schedule FROM service_providers WHERE business_name = ? ORDER BY rating DESC",
            (state.get("business_name"),)
        ).fetchall()

        # Filter by name if the router extracted one
        provider_name = (
            state.get("provider_name")
            or kwargs.get("provider_name")
            or ""
        ).strip().lower()

        if provider_name:
            # Fuzzy match: any part of the search name appears in the provider name
            search_parts = provider_name.replace("dr.", "").replace("dr ", "").split()
            matched = [
                p for p in all_providers
                if any(part in p["name"].lower() for part in search_parts if len(part) > 1)
            ]
        else:
            matched = list(all_providers)

        # Build per-provider availability with their schedule days
        providers_out = []
        for p in matched:
            info = {
                "name": p["name"],
                "specialty": p["specialty"],
                "rating": p["rating"],
                "available": bool(p["available"]),
            }
            # Parse schedule to show working days
            if p["schedule"]:
                try:
                    sched = json.loads(p["schedule"])
                    day_map = {"mon": "Mon", "tue": "Tue", "wed": "Wed",
                               "thu": "Thu", "fri": "Fri", "sat": "Sat", "sun": "Sun"}
                    info["working_days"] = [day_map.get(d, d) for d in sched.keys()]
                    info["hours"] = sched
                except Exception:
                    pass
            providers_out.append(info)

        if not providers_out and provider_name:
            # No match found — return all with a helpful note
            return {
                "success": True,
                "providers": [{"name": p["name"], "specialty": p["specialty"],
                               "rating": p["rating"], "available": bool(p["available"])}
                              for p in all_providers],
                "total_providers": len(all_providers),
                "searched_for": provider_name,
                "message": (
                    f"No provider found matching '{provider_name}'. "
                    f"Here are all {len(all_providers)} staff members available."
                ),
            }

        name_label = f" matching '{provider_name}'" if provider_name else ""
        return {
            "success": True,
            "providers": providers_out,
            "total_providers": len(providers_out),
            "searched_for": provider_name,
            "message": (
                f"Found {len(providers_out)} provider(s){name_label}."
                if providers_out else "No providers found."
            ),
        }
    finally:
        conn.close()


def get_faqs(state: dict, **kwargs) -> dict:
    """Retrieve frequently asked questions from the knowledge base."""
    conn = _db()
    try:
        faqs = conn.execute(
            """SELECT topic, content FROM enriched_knowledge
               WHERE LOWER(topic) LIKE '%faq%' OR LOWER(topic) LIKE '%question%' OR LOWER(topic) LIKE '%common%'
               LIMIT 3"""
        ).fetchall()
        if not faqs:
            faqs = conn.execute("SELECT topic, content FROM enriched_knowledge LIMIT 3").fetchall()
        return {"success": True, "faqs": [dict(f) for f in faqs], "message": f"Retrieved {len(faqs)} FAQ entries."}
    finally:
        conn.close()


def get_promotions(state: dict, **kwargs) -> dict:
    """List current deals, promotions, and special offers."""
    conn = _db()
    try:
        promos = conn.execute(
            """SELECT topic, content FROM enriched_knowledge
               WHERE LOWER(topic) LIKE '%promo%' OR LOWER(topic) LIKE '%deal%'
                  OR LOWER(topic) LIKE '%offer%' OR LOWER(topic) LIKE '%discount%'
                  OR LOWER(topic) LIKE '%seasonal%'
               LIMIT 3"""
        ).fetchall()
        if not promos:
            promos = conn.execute("SELECT topic, content FROM enriched_knowledge LIMIT 2").fetchall()
        return {"success": True, "promotions": [dict(p) for p in promos], "message": f"Found {len(promos)} active promotions."}
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Loyalty Tools
# ─────────────────────────────────────────────

def apply_loyalty_discount(state: dict, **kwargs) -> dict:
    """Apply loyalty points as a discount on the next booking."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        loyalty = conn.execute("SELECT * FROM loyalty_points WHERE user_id = ? AND business_name = ?", (user_id, state.get("business_name"))).fetchone()
        if not loyalty or loyalty["points"] < 100:
            return {
                "success": False,
                "message": "You need at least 100 loyalty points to apply a discount.",
                "current_points": loyalty["points"] if loyalty else 0,
            }
        discount = (loyalty["points"] // 100) * 5
        points_used = (loyalty["points"] // 100) * 100
        new_points = loyalty["points"] - points_used
        conn.execute(
            "UPDATE loyalty_points SET points = ?, updated_at = ? WHERE user_id = ? AND business_name = ?",
            (new_points, datetime.now(timezone.utc).isoformat(), user_id, state.get("business_name")),
        )
        log_db_write(state.get("conversation_id", ""), "loyalty_points", "UPDATE", None, {"action": "redeem", "points_used": points_used, "remaining": new_points})
        conn.commit()
        return {
            "success": True, "discount_applied": discount, "points_used": points_used,
            "remaining_points": new_points,
            "message": f"${discount:.2f} discount applied using {points_used} loyalty points. Remaining: {new_points} points.",
        }
    finally:
        conn.close()


def get_loyalty_balance(state: dict, **kwargs) -> dict:
    """Show the customer's current loyalty points and tier."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        loyalty = conn.execute("SELECT * FROM loyalty_points WHERE user_id = ? AND business_name = ?", (user_id, state.get("business_name"))).fetchone()
        if not loyalty:
            return {"success": True, "points": 0, "tier": "bronze", "message": "No loyalty points yet."}
        points = loyalty["points"]
        tier = loyalty["tier"]
        next_tier = "silver" if tier == "bronze" else "gold" if tier == "silver" else "platinum"
        next_threshold = 500 if tier == "bronze" else 1000 if tier == "silver" else 2000
        return {
            "success": True, "points": points, "tier": tier, "next_tier": next_tier,
            "points_to_next_tier": max(0, next_threshold - points),
            "dollar_value": (points // 100) * 5,
            "message": f"You have {points} loyalty points ({tier} tier). ${(points // 100) * 5:.2f} redeemable.",
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Dispute / Complaint Tool
# ─────────────────────────────────────────────

def handle_dispute(state: dict, **kwargs) -> dict:
    """Handle order disputes, complaints, and discrepancies (e.g. wrong quantity delivered)."""
    user_id = state.get("user_id")
    query = state.get("query", "")
    conn = _db()
    try:
        recent_orders = conn.execute(
            """SELECT o.id, o.items_json, o.total_price, o.status, s.name as service_name
               FROM orders o LEFT JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? AND s.business_name = ? ORDER BY o.scheduled_at DESC LIMIT 5""",
            (user_id, state.get("business_name")),
        ).fetchall()

        orders_text = "\n".join(
            f"Order #{o['id']}: {o['service_name']} | ${o['total_price']} | Status: {o['status']} | Items: {o['items_json'] or 'N/A'}"
            for o in recent_orders
        )

        parse_prompt = f"""Customer complaint: "{query}"
Recent orders:
{orders_text}

Analyse the complaint and return JSON:
{{
  "order_id": 1,
  "complaint_type": "wrong_quantity|missing_item|wrong_item|quality|other",
  "description": "Customer ordered 3 but received 2",
  "suggested_resolution": "Refund for 1 item or redeliver",
  "understood": true
}}
No markdown, just JSON."""

        raw = llm_call(parse_prompt, temperature=0.1, max_tokens=250)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            dispute = json.loads(raw)
        except json.JSONDecodeError:
            return {"success": False, "message": "I'm sorry, I couldn't process the details of your complaint. Could you please describe it again?"}

        # Log the complaint to the DB
        cur = conn.execute(
            """INSERT INTO complaints (user_id, order_id, conversation_id, complaint_type, description, suggested_resolution)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                dispute.get("order_id"),
                state.get("conversation_id"),
                dispute.get("complaint_type", "other"),
                dispute.get("description", query),
                dispute.get("suggested_resolution", "Review needed")
            )
        )
        complaint_id = cur.lastrowid
        log_db_write(state.get("conversation_id", ""), "complaints", "INSERT", complaint_id, {"type": dispute.get("complaint_type"), "order_id": dispute.get("order_id")})
        conn.commit()
        
        ref_id = _generate_ref("REF", complaint_id)

        return {
            "success": True,
            "complaint_id": complaint_id,
            "complaint_ref": ref_id,
            "details": dispute,
            "message": f"I've logged your concern regarding Order #{dispute.get('order_id') or 'unknown'}. Your reference number is {ref_id}. Our team will review this and I'll make sure we address it right away.",
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Profile & Other Tools
# ─────────────────────────────────────────────

def search_web(state: dict, **kwargs) -> dict:
    """Search the web for real-time information using DuckDuckGo."""
    query = state.get("query", "")
    business_name = state.get("business_name") or get_business_meta("business_name", state.get("business_name","")) or "business"
    search_query = f"{business_name} {query}"
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=3))
        formatted = [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ]
        return {"success": True, "results": formatted, "query": search_query, "message": f"Found {len(formatted)} web results."}
    except Exception as e:
        return {"success": False, "message": f"Web search failed: {str(e)}", "results": []}


def escalate_to_human(state: dict, **kwargs) -> dict:
    """Flag the conversation for human agent handoff."""
    conv_id = state.get("conversation_id", "unknown")
    log_agent_event(conv_id, "human_escalation", {
        "user_id": state.get("user_id"), "query": state.get("query", ""),
        "reason": "Customer requested human agent or issue unresolved",
    })
    return {
        "success": True, "escalated": True,
        "message": "Your request has been flagged for a human agent. We will contact you within 24 hours via email.",
    }


def get_global_stats_tool(state: dict, **kwargs) -> dict:
    """Retrieve aggregated business statistics."""
    stats = get_global_stats()
    return {
        "success": True, "stats": stats,
        "message": f"Business has {stats.get('total_customers',0)} customers and ${stats.get('total_revenue',0):.2f} total revenue.",
    }


def update_customer_profile(state: dict, **kwargs) -> dict:
    """Update the customer's profile information (name, phone, address)."""
    user_id = state.get("user_id")
    query = state.get("query", "")
    conn = _db()
    try:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return {"success": False, "message": "User not found."}

        # Use LLM to extract what they want to update
        parse_prompt = f"""Customer says: "{query}"
Current profile: name={user['full_name']}, phone={user['phone']}, email={user['email']}, address={user['address'] or 'not set'}

What field do they want to update and to what value?
Return JSON: {{"field": "phone", "new_value": "555-1234", "understood": true}}
If unclear: {{"understood": false, "message": "Which field?"}}
No markdown, just JSON."""

        raw = llm_call(parse_prompt, temperature=0.1, max_tokens=150)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            update = json.loads(raw)
        except json.JSONDecodeError:
            update = {"understood": False}

        if update.get("understood", False):
            field = update.get("field", "")
            valid_fields = {"full_name": "full_name", "name": "full_name", "phone": "phone", "email": "email", "address": "address"}
            db_field = valid_fields.get(field.lower())
            if db_field:
                conn.execute(f"UPDATE users SET {db_field} = ? WHERE id = ?", (update["new_value"], user_id))
                log_db_write(state.get("conversation_id", ""), "users", "UPDATE", user_id, {"field": db_field, "new_value": update["new_value"]})
                conn.commit()
                return {
                    "success": True,
                    "updated_field": db_field,
                    "new_value": update["new_value"],
                    "message": f"Updated your {field} to: {update['new_value']}",
                }

        return {
            "success": True, "user_id": user_id,
            "current_profile": {"full_name": user["full_name"], "email": user["email"], "phone": user["phone"], "address": user["address"]},
            "message": "To update your profile, type the field and new value. Example: **phone 0555-1234** or **email new@email.com** or **name John Smith**",
        }
    finally:
        conn.close()


def check_family_members(state: dict, **kwargs) -> dict:
    """Check customer's family members for cross-selling opportunities."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        user = conn.execute("SELECT full_name, family_members FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user or not user["family_members"]:
            return {"success": True, "family": [], "message": "No family members on file."}

        try:
            family = json.loads(user["family_members"])
        except json.JSONDecodeError:
            family = []

        # Check last visits for family members
        family_info = []
        for member in family:
            family_info.append({
                "name": member.get("name", ""),
                "relation": member.get("relation", ""),
            })

        return {
            "success": True,
            "customer_name": user["full_name"],
            "family": family_info,
            "message": f"Found {len(family_info)} family member(s) on file: {', '.join(m['name'] for m in family_info)}.",
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Calendar / ICS Helper
# ─────────────────────────────────────────────

def _generate_ics(
    order_id: int,
    title: str,
    start: datetime,
    end: datetime,
    provider: str,
    patient_name: str = "Customer",
    business_name: str = "",
    method: str = "REQUEST",   # "REQUEST" = book/update, "CANCEL" = remove from calendar
    sequence: int = 0,         # increment on every update/cancel so calendar app knows it's newer
) -> str:
    """
    Generate a standards-compliant .ics calendar file.

    Reschedule flow  → call twice:
        1st  _generate_ics(..., method="CANCEL",  sequence=prev+1)  → removes old event
        2nd  _generate_ics(..., method="REQUEST", sequence=prev+2)  → adds new event

    Cancel flow      → call once:
             _generate_ics(..., method="CANCEL",  sequence=1)

    The UID is always  booking-{order_id}@genericagent  so the calendar app
    can match updates/cancellations to the original event.
    """
    # Use project root so calendar/ is always in the project folder
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    calendar_dir = os.path.join(_project_root, "calendar")
    os.makedirs(calendar_dir, exist_ok=True)

    safe_name    = re.sub(r"[^a-z0-9_]", "_", patient_name.lower())
    safe_method  = method.upper()

    # Separate filenames: cancel gets its own file so both are importable
    if safe_method == "CANCEL":
        filename = f"{safe_name}_cancel_{order_id}.ics"
    else:
        filename = f"{safe_name}_appointment_{order_id}.ics"

    filepath = os.path.join(calendar_dir, filename)

    # Times — handle both naive and aware datetimes
    try:
        start_utc = start.astimezone(timezone.utc)
        end_utc   = end.astimezone(timezone.utc)
    except (TypeError, AttributeError):
        start_utc = start.replace(tzinfo=timezone.utc)
        end_utc   = end.replace(tzinfo=timezone.utc)

    dtstamp   = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dtstart   = start_utc.strftime('%Y%m%dT%H%M%SZ')
    dtend     = end_utc.strftime('%Y%m%dT%H%M%SZ')
    dur_min   = int((end - start).total_seconds() / 60)
    ref_num   = f"APPT-{order_id:05d}"
    safe_prov = provider or "Assigned Staff"
    biz_label = business_name or "Business"

    # ── Friendly date/time strings for DESCRIPTION ────────────────────────────
    start_local = start if start.tzinfo is None else start.astimezone()
    end_local   = end   if end.tzinfo   is None else end.astimezone()
    date_str  = start_local.strftime("%A, %B %d, %Y")         # e.g. Tuesday, March 17, 2026
    time_from = start_local.strftime("%I:%M %p").lstrip("0")  # e.g. 9:00 AM
    time_to   = end_local.strftime("%I:%M %p").lstrip("0")    # e.g. 10:00 AM

    # ── SUMMARY line ──────────────────────────────────────────────────────────
    if safe_method == "CANCEL":
        summary = f"CANCELLED: {title} — {patient_name} with {safe_prov}"
        status  = "CANCELLED"
    else:
        summary = f"{title} — {patient_name} with {safe_prov}"
        status  = "CONFIRMED"

    # ── Structured DESCRIPTION (readable in any calendar app) ─────────────────
    sep = "─" * 38
    if safe_method == "CANCEL":
        description = (
            f"⚠ APPOINTMENT CANCELLED\\n"
            f"{sep}\\n"
            f"Service:  {title}\\n"
            f"Patient:  {patient_name}\\n"
            f"Provider: {safe_prov}\\n"
            f"Was:      {date_str}  {time_from} – {time_to}\\n"
            f"Ref:      {ref_num}\\n"
            f"{sep}\\n"
            f"This appointment has been cancelled.\\n"
            f"Please contact {biz_label} to rebook."
        )
    else:
        description = (
            f"📋 APPOINTMENT DETAILS\\n"
            f"{sep}\\n"
            f"👤 Patient:   {patient_name}\\n"
            f"🩺 Provider:  {safe_prov}\\n"
            f"💼 Service:   {title}\\n"
            f"📅 Date:      {date_str}\\n"
            f"🕐 Time:      {time_from} – {time_to} ({dur_min} min)\\n"
            f"🏥 Clinic:    {biz_label}\\n"
            f"📋 Reference: {ref_num}\\n"
            f"{sep}\\n"
            f"⏰ Reminders set for 24 hours and 1 hour before."
        )

    # ── Alarms (only for confirmed events) ───────────────────────────────────
    alarms = ""
    if safe_method != "CANCEL":
        alarms = (
            f"BEGIN:VALARM\r\n"
            f"TRIGGER:-PT24H\r\n"
            f"ACTION:DISPLAY\r\n"
            f"DESCRIPTION:Reminder: {title} with {safe_prov} is tomorrow\r\n"
            f"END:VALARM\r\n"
            f"BEGIN:VALARM\r\n"
            f"TRIGGER:-PT1H\r\n"
            f"ACTION:DISPLAY\r\n"
            f"DESCRIPTION:Reminder: {title} starts in 1 hour\r\n"
            f"END:VALARM\r\n"
        )

    ics_content = (
        f"BEGIN:VCALENDAR\r\n"
        f"VERSION:2.0\r\n"
        f"PRODID:-//Generic AI Business Agent//Appointment System//EN\r\n"
        f"CALSCALE:GREGORIAN\r\n"
        f"METHOD:{safe_method}\r\n"
        f"BEGIN:VEVENT\r\n"
        f"UID:booking-{order_id}@genericagent\r\n"
        f"DTSTAMP:{dtstamp}\r\n"
        f"DTSTART:{dtstart}\r\n"
        f"DTEND:{dtend}\r\n"
        f"SUMMARY:{summary}\r\n"
        f"DESCRIPTION:{description}\r\n"
        f"ORGANIZER;CN={biz_label}:mailto:noreply@genericagent.com\r\n"
        f"ATTENDEE;CN={patient_name};RSVP=TRUE:mailto:customer@example.com\r\n"
        f"LOCATION:{biz_label}\r\n"
        f"STATUS:{status}\r\n"
        f"SEQUENCE:{sequence}\r\n"
        f"{alarms}"
        f"END:VEVENT\r\n"
        f"END:VCALENDAR\r\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(ics_content)

    return filepath


def get_calendar(state: dict, **kwargs) -> dict:
    """Get the customer's upcoming calendar events / appointments."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        events = conn.execute(
            """SELECT e.*, s.name as service_name FROM calendar_events e
               JOIN orders o ON e.order_id = o.id
               JOIN services s ON o.service_id = s.id
               WHERE e.user_id = ? AND e.status != 'cancelled' AND s.business_name = ?
               ORDER BY e.start_time ASC LIMIT 10""",
            (user_id, state.get("business_name")),
        ).fetchall()
        return {
            "success": True,
            "events": [dict(e) for e in events],
            "total": len(events),
            "message": f"You have {len(events)} upcoming appointment(s).",
        }
    finally:
        conn.close()


def greet_customer(state: dict, **kwargs) -> dict:
    """
    Greet the customer warmly and introduce the business with its key services,
    hours, and a friendly invitation to help.
    Called when the customer says hello / hi / hey / good morning etc.
    """
    business_name = state.get("business_name", "us")
    business_type = state.get("business_type", "")
    all_chunks    = state.get("all_chunks", [])

    conn = _db()
    try:
        # Grab real services from DB
        services = conn.execute(
            "SELECT name, price FROM services WHERE business_name = ? ORDER BY id LIMIT 8",
            (business_name,)
        ).fetchall()

        # Grab hours from meta
        from core.database import get_business_meta
        hours = get_business_meta("business_hours", business_name) or ""

        # Build service list for greeting
        svc_lines = ""
        if services:
            svc_lines = "\n".join(f"  • {s['name']} — ${s['price']:.2f}" for s in services)

        hours_line = f"\nOur hours: {hours}" if hours else ""

        greeting_data = {
            "business_name": business_name,
            "business_type": business_type,
            "services_snippet": svc_lines,
            "hours": hours,
        }

        return {
            "success": True,
            "greeting_data": greeting_data,
            "message": (
                f"Hello and welcome to {business_name}!\n\n"
                f"Here's a quick look at what we offer:\n{svc_lines}\n"
                f"{hours_line}\n\n"
                "I'm here to help you book an appointment, place an order, or answer any questions. "
                "What can I do for you today? 😊"
            ),
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# TOOLS REGISTRY
# ─────────────────────────────────────────────

TOOLS: dict[str, tuple] = {
    # Greeting
    "greet_customer": (greet_customer, "Activates when the customer says hello, hi, hey, good morning, good afternoon, or any greeting. Gives a warm welcome with services overview."),

    # Cart / Order management
    "add_to_cart": (add_to_cart, "Activates when the customer wants to order, add, or get an item/product/service. Also when they say 'I want X' or 'give me X'."),
    "create_order": (add_to_cart, "Alias: Activates when the customer wants to create or start an order. Same as add_to_cart."),
    "remove_from_cart": (remove_from_cart, "Activates when the customer wants to remove, delete, or change their mind about an item in their order/cart. Also 'I don't want X anymore'."),
    "update_order": (add_to_cart, "Alias: Activates when the customer wants to update, change, or modify their order (add/remove items). Use add_to_cart for adds, remove_from_cart for removals."),
    "view_cart": (view_cart, "Activates when the customer asks to see their cart, current order, what they've ordered so far."),
    "confirm_order": (confirm_order, "Activates when the customer says 'confirm', 'yes', 'place order', 'that's all', 'finalize', or agrees to a final order summary."),

    # Booking / Scheduling
    "book_appointment": (book_appointment, "Activates when the customer wants to book/schedule AND has explicitly provided a date and time (e.g. 'tomorrow at 2pm', 'Monday 3pm'). Do NOT call if they only said which service they want (e.g. 'I want Men's haircut') without date and time — ask for date/time first."),
    "reschedule_booking": (reschedule_booking, "Activates when the customer wants to change, move, or reschedule the date or time of an existing booking."),
    "modify_appointment": (reschedule_booking, "Alias: Activates when the customer wants to modify or change an existing appointment. Same as reschedule_booking."),
    "cancel_booking": (cancel_booking, "Activates when the customer wants to cancel, delete, or remove an existing booking or reservation."),
    "cancel_appointment": (cancel_booking, "Alias: Activates when the customer wants to cancel an appointment. Same as cancel_booking."),
    "check_availability": (check_availability, "Activates when the customer asks about availability, open slots, time slots, when they can come, or 'when is X available'."),

    # Delivery / Address
    "set_delivery_type": (set_delivery_type, "Activates when the customer mentions pickup, delivery, or is asked about how they want to receive their order."),
    "validate_address": (validate_address, "Activates when the customer provides a delivery address, street, postal code, or when address validation is needed."),
    "set_delivery_address": (set_delivery_address, "Activates when the customer explicitly provides or confirms a delivery address with postal code. Validates and saves it."),

    # Information
    "search_knowledge": (search_knowledge, "Activates when the customer asks a question that requires looking up business info, policies, menu, services, or knowledge base content."),
    "get_business_info": (get_business_info, "Activates when the customer asks about the business in general, hours, staff, location, or 'tell me about you'."),
    "get_pricing": (get_pricing, "Activates when the customer asks about prices, costs, fees, rates, or how much something costs."),
    "get_recommendations": (get_recommendations, "Activates when the customer asks for recommendations, suggestions, what to try, or what is popular. Also for upselling."),
    "get_order_history": (get_order_history, "Activates when the customer asks about their past orders, previous bookings, or purchase history."),
    "get_business_hours": (get_business_hours, "Activates when the customer asks about opening hours, closing time, when the business is open, holiday hours."),
    "get_provider_info": (get_provider_info, "Activates when the customer asks about staff, therapists, doctors, providers, or who will serve them."),
    "get_faqs": (get_faqs, "Activates when the customer asks a general question, how something works, policies, or requests FAQs."),
    "get_promotions": (get_promotions, "Activates when the customer asks about deals, promotions, discounts, offers, or special prices."),

    # Loyalty
    "apply_loyalty_discount": (apply_loyalty_discount, "Activates when the customer wants to redeem, use, or apply loyalty points or get a discount via points."),
    "get_loyalty_balance": (get_loyalty_balance, "Activates when the customer asks about their loyalty points, rewards balance, tier, or membership status."),

    # Dispute
    "handle_dispute": (handle_dispute, "Activates when the customer has a complaint, dispute, wrong order, missing item, quality issue, or says something was wrong."),

    # Profile & Family
    "update_customer_profile": (update_customer_profile, "Activates when the customer wants to update their name, phone, email, address, or personal details."),
    "check_family_members": (check_family_members, "Activates when recommending services for family, asking about family members, or cross-selling for spouse/children."),

    # Calendar
    "get_calendar": (get_calendar, "Activates when the customer asks about their upcoming appointments, schedule, or calendar."),

    # Web search
    "search_web": (search_web, "Activates when the customer asks for current news, trends, external information, or anything not in the knowledge base."),

    # Escalation
    "escalate_to_human": (escalate_to_human, "Activates when the customer is frustrated, requests a human agent, or when the AI cannot resolve the issue."),

    # Stats
    "get_global_stats": (get_global_stats_tool, "Activates when asked about overall business performance, statistics, or revenue (usually staff/admin queries)."),
}