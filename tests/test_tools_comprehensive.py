"""
test_tools_comprehensive.py
===========================
Direct unit tests for agent tools. Uses a fixture DB and calls tools with
prepared state — no LLM routing. Tests pass reliably without API calls for
view_cart, confirm_order, get_order_history. add_to_cart uses LLM internally.

Run: python -m pytest tests/test_tools_comprehensive.py -v
Or:  python tests/test_tools_comprehensive.py
"""

import os
import sys
import sqlite3
import json
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Use a dedicated test DB
TEST_DB = "test_tools_comprehensive.db"


def _setup_fixture_db():
    """Create test DB with schema, services, and a user."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    os.environ["DB_NAME"] = TEST_DB

    from core.database import init_db

    init_db()

    conn = sqlite3.connect(TEST_DB)
    conn.row_factory = sqlite3.Row
    try:
        # Insert user
        conn.execute(
            """INSERT INTO users (username, email, password_hash, full_name)
               VALUES (?, ?, ?, ?)""",
            ("testuser", "test@example.com", "hash", "Test User"),
        )
        user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Insert services for Mario Pizza
        services = [
            ("Mario's Italiano Pizzeria", "Garlic Bread", 4.99, "Appetizer"),
            ("Mario's Italiano Pizzeria", "Gelato (2 scoops)", 5.99, "Dessert"),
            ("Mario's Italiano Pizzeria", "Lasagna", 16.99, "Main"),
        ]
        for bname, name, price, cat in services:
            conn.execute(
                """INSERT INTO services (business_name, name, price, category)
                   VALUES (?, ?, ?, ?)""",
                (bname, name, price, cat),
            )
        conn.commit()
    finally:
        conn.close()

    return user_id


def _get_service_ids(business_name: str):
    conn = sqlite3.connect(TEST_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, name, price FROM services WHERE business_name = ?",
        (business_name,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _insert_cart(user_id: int, session_id: str, business_name: str, items: list):
    """Insert cart items directly. items: [(service_id, service_name, qty, unit_price), ...]"""
    conn = sqlite3.connect(TEST_DB)
    conn.row_factory = sqlite3.Row
    try:
        for sid, sname, qty, price in items:
            conn.execute(
                """INSERT INTO cart (user_id, session_id, business_name, service_id, service_name, quantity, unit_price)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, session_id, business_name, sid, sname, qty, price),
            )
        conn.commit()
    finally:
        conn.close()


def _clear_cart(user_id: int, session_id: str, business_name: str):
    conn = sqlite3.connect(TEST_DB)
    conn.execute(
        "DELETE FROM cart WHERE user_id = ? AND session_id = ? AND business_name = ?",
        (user_id, session_id, business_name),
    )
    conn.commit()
    conn.close()


def test_view_cart_empty():
    """view_cart with empty cart returns empty message."""
    user_id = _setup_fixture_db()
    session_id = str(uuid.uuid4())
    state = {
        "user_id": user_id,
        "conversation_id": session_id,
        "business_name": "Mario's Italiano Pizzeria",
    }

    from agent.tools import view_cart

    result = view_cart(state)
    assert result.get("success") is True
    assert result.get("cart_items") == []
    assert result.get("cart_total") == 0
    assert "empty" in result.get("message", "").lower()


def test_view_cart_with_items():
    """view_cart with items returns correct cart_items and cart_total."""
    user_id = _setup_fixture_db()
    session_id = str(uuid.uuid4())
    svcs = _get_service_ids("Mario's Italiano Pizzeria")
    assert len(svcs) >= 2
    _insert_cart(
        user_id,
        session_id,
        "Mario's Italiano Pizzeria",
        [
            (svcs[0]["id"], svcs[0]["name"], 1, svcs[0]["price"]),
            (svcs[1]["id"], svcs[1]["name"], 2, svcs[1]["price"]),
        ],
    )

    state = {
        "user_id": user_id,
        "conversation_id": session_id,
        "business_name": "Mario's Italiano Pizzeria",
    }

    from agent.tools import view_cart

    result = view_cart(state)
    assert result.get("success") is True
    items = result.get("cart_items", [])
    assert len(items) == 2
    total = result.get("cart_total", 0)
    expected = svcs[0]["price"] * 1 + svcs[1]["price"] * 2
    assert abs(total - expected) < 0.01


def test_confirm_order_restaurant():
    """confirm_order creates order, clears cart, returns items/total and cart_items=[], cart_total=0."""
    user_id = _setup_fixture_db()
    session_id = str(uuid.uuid4())
    svcs = _get_service_ids("Mario's Italiano Pizzeria")
    _insert_cart(
        user_id,
        session_id,
        "Mario's Italiano Pizzeria",
        [
            (svcs[0]["id"], svcs[0]["name"], 1, svcs[0]["price"]),
            (svcs[1]["id"], svcs[1]["name"], 1, svcs[1]["price"]),
        ],
    )

    state = {
        "user_id": user_id,
        "conversation_id": session_id,
        "business_name": "Mario's Italiano Pizzeria",
        "business_type": "restaurant",
    }

    from agent.tools import confirm_order

    result = confirm_order(state)
    assert result.get("success") is True
    assert "order_id" in result
    assert "order_ref" in result
    assert result.get("cart_items") == []
    assert result.get("cart_total") == 0
    items = result.get("items", [])
    assert len(items) == 2
    total = result.get("total", 0)
    expected = svcs[0]["price"] + svcs[1]["price"]
    assert abs(total - expected) < 0.01

    # Cart should be empty
    from agent.tools import view_cart

    view_result = view_cart(state)
    assert view_result.get("cart_items") == []


def test_confirm_order_appointment_business_rejected():
    """confirm_order for dental/clinic returns needs_booking, does not create order."""
    user_id = _setup_fixture_db()
    session_id = str(uuid.uuid4())
    svcs = _get_service_ids("Mario's Italiano Pizzeria")
    _insert_cart(
        user_id,
        session_id,
        "Mario's Italiano Pizzeria",
        [(svcs[0]["id"], svcs[0]["name"], 1, svcs[0]["price"])],
    )

    state = {
        "user_id": user_id,
        "conversation_id": session_id,
        "business_name": "Mario's Italiano Pizzeria",
        "business_type": "dental",
    }

    from agent.tools import confirm_order

    result = confirm_order(state)
    assert result.get("success") is False
    assert result.get("needs_booking") is True


def test_get_order_history_after_confirm():
    """get_order_history returns confirmed order with cost breakdown."""
    user_id = _setup_fixture_db()
    session_id = str(uuid.uuid4())
    svcs = _get_service_ids("Mario's Italiano Pizzeria")
    _insert_cart(
        user_id,
        session_id,
        "Mario's Italiano Pizzeria",
        [(svcs[0]["id"], svcs[0]["name"], 1, svcs[0]["price"])],
    )

    state = {
        "user_id": user_id,
        "conversation_id": session_id,
        "business_name": "Mario's Italiano Pizzeria",
        "business_type": "restaurant",
    }

    from agent.tools import confirm_order, get_order_history

    confirm_order(state)
    result = get_order_history(state)
    assert result.get("success") is True
    orders = result.get("orders", [])
    assert len(orders) >= 1
    o = orders[0]
    assert o.get("status") == "confirmed"
    assert "total" in o or "total_price" in o


def run_standalone():
    """Run tests without pytest."""
    print("Running direct tool tests...")
    test_view_cart_empty()
    print("  [OK] view_cart empty")
    test_view_cart_with_items()
    print("  [OK] view_cart with items")
    test_confirm_order_restaurant()
    print("  [OK] confirm_order restaurant")
    test_confirm_order_appointment_business_rejected()
    print("  [OK] confirm_order appointment rejected")
    test_get_order_history_after_confirm()
    print("  [OK] get_order_history")
    print("All tests passed.")
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


if __name__ == "__main__":
    run_standalone()
