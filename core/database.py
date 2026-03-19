"""
core/database.py
----------------
SQLite schema creation and synthetic data generation.
Includes: cart system, address fields, calendar events.
"""

import sqlite3
import os
import json
import random
import bcrypt
from datetime import datetime, timezone, timedelta
from faker import Faker
from core.llm_client import llm_call

fake = Faker()


SCHEMA_SQL = """
-- Users (customers who can log in)
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    UNIQUE NOT NULL,
    email       TEXT    UNIQUE NOT NULL,
    password_hash TEXT  NOT NULL,
    full_name   TEXT,
    phone       TEXT,
    address     TEXT,
    postal_code TEXT,
    city        TEXT,
    family_members TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- Service providers / staff
CREATE TABLE IF NOT EXISTS service_providers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name TEXT,
    name        TEXT    NOT NULL,
    specialty   TEXT,
    rating      REAL    DEFAULT 4.5,
    available   INTEGER DEFAULT 1,
    schedule    TEXT
);

-- Services / products offered by the business
CREATE TABLE IF NOT EXISTS services (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name TEXT,
    name        TEXT    NOT NULL,
    description TEXT,
    price       REAL    NOT NULL,
    duration_min INTEGER,
    category    TEXT,
    modifiers   TEXT
);

-- Shopping cart (active order being built)
CREATE TABLE IF NOT EXISTS cart (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name TEXT,
    user_id     INTEGER REFERENCES users(id),
    session_id  TEXT    NOT NULL,
    service_id  INTEGER REFERENCES services(id),
    service_name TEXT,
    quantity    INTEGER DEFAULT 1,
    unit_price  REAL,
    modifiers   TEXT,
    notes       TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- Confirmed orders
CREATE TABLE IF NOT EXISTS orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name   TEXT,
    user_id         INTEGER REFERENCES users(id),
    service_id      INTEGER REFERENCES services(id),
    provider_id     INTEGER REFERENCES service_providers(id),
    status          TEXT    DEFAULT 'completed',
    order_type      TEXT    DEFAULT 'order',
    scheduled_at    TEXT,
    completed_at    TEXT,
    notes           TEXT,
    total_price     REAL,
    delivery_type   TEXT,
    delivery_address TEXT,
    delivery_postal TEXT,
    items_json      TEXT
);

-- Calendar events (local .ics compatible)
CREATE TABLE IF NOT EXISTS calendar_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name TEXT,
    user_id     INTEGER REFERENCES users(id),
    order_id    INTEGER REFERENCES orders(id),
    title       TEXT    NOT NULL,
    description TEXT,
    start_time  TEXT    NOT NULL,
    end_time    TEXT    NOT NULL,
    location    TEXT,
    provider    TEXT,
    status      TEXT    DEFAULT 'confirmed',
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- Loyalty programme
CREATE TABLE IF NOT EXISTS loyalty_points (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name TEXT,
    user_id     INTEGER REFERENCES users(id),
    points      INTEGER DEFAULT 0,
    tier        TEXT    DEFAULT 'bronze',
    updated_at  TEXT    DEFAULT (datetime('now')),
    UNIQUE(user_id, business_name)
);

-- PDF knowledge chunks
CREATE TABLE IF NOT EXISTS pdf_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name   TEXT,
    chunk_index     INTEGER,
    page_num        INTEGER,
    section_title   TEXT,
    text            TEXT    NOT NULL,
    topic_tags      TEXT,
    created_at      TEXT    DEFAULT (datetime('now'))
);

-- LLM-enriched supplementary knowledge
CREATE TABLE IF NOT EXISTS enriched_knowledge (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name TEXT,
    topic       TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    source      TEXT    DEFAULT 'llm_enrichment',
    topic_tags  TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- Customer complaints and disputes
CREATE TABLE IF NOT EXISTS complaints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER REFERENCES users(id),
    order_id        INTEGER REFERENCES orders(id),
    complaint_type  TEXT,
    description     TEXT,
    suggested_resolution TEXT,
    status          TEXT    DEFAULT 'open',
    created_at      TEXT    DEFAULT (datetime('now'))
);

-- Chat messages
CREATE TABLE IF NOT EXISTS conversations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name   TEXT,
    conversation_id TEXT    NOT NULL,
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL
);

-- Tool call log
CREATE TABLE IF NOT EXISTS tool_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL,
    tool_name       TEXT    NOT NULL,
    inputs          TEXT,
    outputs         TEXT,
    activated       INTEGER DEFAULT 1,
    timestamp       TEXT    NOT NULL
);

-- Generic agent events
CREATE TABLE IF NOT EXISTS agent_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,
    details         TEXT,
    timestamp       TEXT    NOT NULL
);

-- Business metadata
CREATE TABLE IF NOT EXISTS business_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT
);

-- Conversation state (for multi-turn slot filling)
CREATE TABLE IF NOT EXISTS conversation_state (
    conversation_id TEXT PRIMARY KEY,
    state_json      TEXT NOT NULL,
    updated_at      TEXT DEFAULT (datetime('now'))
);
"""


def init_db() -> None:
    conn = _get_db()
    try:
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency: multiple readers + one writer
        conn.executescript(SCHEMA_SQL)
        # Migrations for older DBs (add columns if missing)
        for table, col in [("conversations", "business_name"), ("enriched_knowledge", "business_name")]:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.commit()
    finally:
        conn.close()


def _get_db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def set_business_meta(key: str, value: str, business_name: str = "") -> None:
    """Store a metadata value scoped to a specific business."""
    conn = _get_db()
    try:
        scoped_key = f"{business_name}::{key}" if business_name else key
        conn.execute(
            "INSERT OR REPLACE INTO business_meta (key, value) VALUES (?, ?)",
            (scoped_key, value),
        )
        conn.commit()
    finally:
        conn.close()


def get_business_meta(key: str, business_name: str = "") -> str | None:
    """Retrieve a metadata value scoped to a specific business."""
    conn = _get_db()
    try:
        scoped_key = f"{business_name}::{key}" if business_name else key
        row = conn.execute(
            "SELECT value FROM business_meta WHERE key = ?", (scoped_key,)
        ).fetchone()
        # Fallback: try unscoped key for backwards compatibility
        if row is None and business_name:
            row = conn.execute(
                "SELECT value FROM business_meta WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None
    finally:
        conn.close()


def list_businesses() -> list[dict]:
    """
    Return all businesses that have been loaded into the database,
    ordered by most recently added first.
    """
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT DISTINCT business_name,
                      MAX(created_at) as last_loaded,
                      COUNT(*) as chunk_count
               FROM pdf_chunks
               WHERE business_name IS NOT NULL AND business_name != ''
               GROUP BY business_name
               ORDER BY last_loaded DESC"""
        ).fetchall()
        return [dict(r) for r in rows] if rows else []
    except Exception:
        return []
    finally:
        conn.close()


def business_is_loaded(business_name: str) -> bool:
    """Check whether a specific business has already been processed and stored."""
    conn = _get_db()
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM pdf_chunks WHERE business_name = ?",
            (business_name,)
        ).fetchone()[0]
        return count > 0
    except Exception:
        return False
    finally:
        conn.close()


def migrate_add_business_name(business_name: str) -> None:
    """
    Migration helper: add business_name to existing orders and calendar_events rows
    that were inserted before the column existed (SQLite ADD COLUMN is safe to run
    multiple times because we use IF NOT EXISTS equivalent try/except).
    Also back-fills NULL business_name rows using the service join.
    """
    conn = _get_db()
    try:
        # Add columns if they don't exist yet (SQLite doesn't support IF NOT EXISTS on ADD COLUMN)
        for table, col in [("orders", "business_name"), ("calendar_events", "business_name"),
                            ("complaints", "business_name")]:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
                conn.commit()
            except Exception:
                pass  # Column already exists

        # Back-fill orders.business_name from joined services
        conn.execute(
            """UPDATE orders SET business_name = ?
               WHERE business_name IS NULL
                 AND service_id IN (SELECT id FROM services WHERE business_name = ?)""",
            (business_name, business_name)
        )
        # Back-fill calendar_events.business_name via order join
        conn.execute(
            """UPDATE calendar_events SET business_name = ?
               WHERE business_name IS NULL
                 AND order_id IN (SELECT id FROM orders WHERE business_name = ?)""",
            (business_name, business_name)
        )
        conn.commit()
    finally:
        conn.close()


def _already_seeded(business_name: str) -> bool:
    conn = _get_db()
    try:
        key = f"seeded_{business_name.lower().replace(' ', '_')}"
        row = conn.execute(
            "SELECT value FROM business_meta WHERE key = ?", (key,)
        ).fetchone()
        return row is not None and row["value"] == "1"
    finally:
        conn.close()


# ── Conversation state persistence ──────────────

def save_conversation_state(conversation_id: str, state: dict) -> None:
    conn = _get_db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO conversation_state (conversation_id, state_json, updated_at) VALUES (?, ?, ?)",
            (conversation_id, json.dumps(state, default=str), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def load_conversation_state(conversation_id: str) -> dict | None:
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT state_json FROM conversation_state WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return json.loads(row["state_json"]) if row else None
    finally:
        conn.close()


# ── Synthetic data generation ──────────────

def _get_named_providers(business_type: str) -> list[dict]:
    """
    Return a fixed set of named service providers based on business type.
    ALL business types always include staff members whose names contain
    "Peter", "Sarah", and "Ana Rodriguez" so smoke-test assertions are
    consistent regardless of which business PDF is loaded.
    """
    bt = business_type.lower()
    full_week = {"mon": "9-17", "tue": "9-17", "wed": "9-17", "thu": "9-17", "fri": "9-17"}
    long_week  = {"mon": "9-20", "tue": "9-20", "wed": "9-20", "thu": "9-20", "fri": "9-20", "sat": "9-17"}
    all_day    = {"mon":"11-23","tue":"11-23","wed":"11-23","thu":"11-23",
                  "fri":"11-23","sat":"11-23","sun":"11-23"}

    if "dental" in bt or "clinic" in bt or "dentist" in bt or "derm" in bt or "medic" in bt:
        return [
            {"name": "Dr. Peter Smith",   "specialty": "General Dentistry",  "schedule": full_week},
            {"name": "Dr. Sarah Chen",    "specialty": "Orthodontics",        "schedule": full_week},
            {"name": "Ana Rodriguez",     "specialty": "Dental Hygienist",    "schedule": {"mon":"9-17","wed":"9-17","fri":"9-17","sat":"9-13"}},
            {"name": "Dr. James Wilson",  "specialty": "Oral Surgery",        "schedule": {"tue":"9-17","thu":"9-17"}},
        ]
    elif "beauty" in bt or "salon" in bt or "spa" in bt or "cosmet" in bt or "hair" in bt or "nail" in bt or "barber" in bt:
        return [
            {"name": "Dr. ABC",           "specialty": "Medical Aesthetician", "schedule": {"wed":"9-17","thu":"9-17","fri":"9-17"}},
            {"name": "Ana Rodriguez",     "specialty": "Hair & Styling",      "schedule": long_week},
            {"name": "Sarah Martinez",    "specialty": "Nail Technician",     "schedule": long_week},
            {"name": "Peter Johnson",     "specialty": "Skin & Facials",      "schedule": {"tue":"9-18","wed":"9-18","thu":"9-18","fri":"9-18","sat":"9-17"}},
            {"name": "Emma Chen",         "specialty": "Waxing & Threading",  "schedule": long_week},
        ]
    elif "photo" in bt or "photography" in bt or "studio" in bt:
        return [
            {"name": "Peter Turner",      "specialty": "Portrait Photography",  "schedule": full_week},
            {"name": "Sarah Kim",         "specialty": "Event Photography",     "schedule": long_week},
            {"name": "Ana Rodriguez",     "specialty": "Studio Coordinator",    "schedule": full_week},
        ]
    elif "clean" in bt or "laundry" in bt or "dry" in bt or "tailor" in bt:
        return [
            {"name": "Peter Mendez",      "specialty": "Dry Cleaning",        "schedule": full_week},
            {"name": "Sarah Park",        "specialty": "Laundry & Pressing",  "schedule": full_week},
            {"name": "Ana Rodriguez",     "specialty": "Alterations",         "schedule": {"mon":"9-17","tue":"9-17","wed":"9-17","thu":"9-17"}},
        ]
    elif "pizza" in bt or "restaurant" in bt or "food" in bt or "cafe" in bt or "bakery" in bt or "diner" in bt:
        return [
            {"name": "Peter Rossi",       "specialty": "Head Chef",           "schedule": all_day},
            {"name": "Sarah Bianchi",     "specialty": "Sous Chef",           "schedule": all_day},
            {"name": "Ana Rodriguez",     "specialty": "Front of House",      "schedule": all_day},
        ]
    elif "gym" in bt or "fitness" in bt or "yoga" in bt or "physio" in bt or "chiro" in bt:
        return [
            {"name": "Peter Williams",    "specialty": "Personal Training",   "schedule": long_week},
            {"name": "Sarah Chen",        "specialty": "Yoga & Pilates",      "schedule": long_week},
            {"name": "Ana Rodriguez",     "specialty": "Physiotherapy",       "schedule": full_week},
        ]
    else:
        # Generic service business — always seed same names
        return [
            {"name": "Peter Johnson",     "specialty": "Senior Specialist",   "schedule": full_week},
            {"name": "Sarah Williams",    "specialty": "General Specialist",  "schedule": long_week},
            {"name": "Ana Rodriguez",     "specialty": "Expert Consultant",   "schedule": full_week},
            {"name": "Marcus Davis",      "specialty": "Junior Specialist",   "schedule": {"tue":"9-17","wed":"9-17","thu":"9-17","fri":"9-17"}},
        ]


def _get_named_clients() -> list[dict]:
    """
    Return a fixed set of named synthetic clients.
    Named clients (Ana, Steven, John, etc.) make conflict scenarios reproducible.
    """
    pw_hash = bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode()
    base_clients = [
        # Named clients — used in conflict demonstration scenarios
        {"username": "ana_thompson",  "full_name": "Ana Thompson",   "email": "ana.thompson@example.com",   "phone": "416-555-0101", "address": "12 King St West", "postal_code": "M5H 1A1", "city": "Toronto",   "family_members": json.dumps([{"relation":"spouse","gender":"Male","name":"Robert"}])},
        {"username": "steven_williams","full_name": "Steven Williams","email": "steven.w@example.com",       "phone": "416-555-0202", "address": "45 Bay Street",   "postal_code": "M5J 2A9", "city": "Toronto",   "family_members": json.dumps([])},
        {"username": "john_carter",   "full_name": "John Carter",    "email": "john.carter@example.com",    "phone": "416-555-0303", "address": "78 Front Street", "postal_code": "M5E 1B4", "city": "Toronto",   "family_members": json.dumps([{"relation":"spouse","gender":"Female","name":"Mary"},{"relation":"child","gender":"Male","name":"Jack"}])},
        {"username": "maria_garcia",  "full_name": "Maria Garcia",   "email": "maria.garcia@example.com",   "phone": "416-555-0404", "address": "22 Queen St E",  "postal_code": "M5C 1R7", "city": "Toronto",   "family_members": json.dumps([{"relation":"child","gender":"Female","name":"Sofia"}])},
        {"username": "hammad_ali",    "full_name": "Hammad Ali",     "email": "hammad.ali@example.com",     "phone": "416-555-0505", "address": "55 Bloor St W",  "postal_code": "M5S 1Y9", "city": "Toronto",   "family_members": json.dumps([{"relation":"spouse","gender":"Female","name":"Fatima"}])},
        {"username": "robert_simpson","full_name": "Robert Simpson", "email": "robert.s@example.com",       "phone": "416-555-0606", "address": "100 Yonge Street","postal_code": "M5C 2W1", "city": "Toronto",   "family_members": json.dumps([{"relation":"spouse","gender":"Female","name":"Linda"},{"relation":"child","gender":"Male","name":"Tom"}])},
        {"username": "linda_white",   "full_name": "Linda White",    "email": "linda.white@example.com",    "phone": "905-555-0707", "address": "34 Oak Avenue",  "postal_code": "L6H 2P4", "city": "Oakville",  "family_members": json.dumps([])},
        {"username": "david_brown",   "full_name": "David Brown",    "email": "david.b@example.com",        "phone": "905-555-0808", "address": "8 Maple Drive",  "postal_code": "L4K 5H7", "city": "Vaughan",   "family_members": json.dumps([{"relation":"child","gender":"Female","name":"Emma"}])},
    ]
    for c in base_clients:
        c["password_hash"] = pw_hash
    return base_clients


def generate_synthetic_data(business_type: str, business_name: str, chunks: list[dict] = None) -> None:
    """
    Seed the database with:
      1. Services/products extracted from the business document by LLM (no hardcoded fallbacks).
      2. Named service providers (e.g. Dr. Peter Smith, Ana Rodriguez).
      3. Named synthetic clients (Ana Thompson, Steven Williams, etc.).
      4. Past historical orders for all clients.
      5. FUTURE confirmed appointments spread across the next 14 days,
         deliberately creating booking conflicts to exercise the
         check_availability / book_appointment overlap logic.
    """
    if _already_seeded(business_name):
        return

    import re as _re

    context_text = ""
    if chunks:
        context_text = "\n".join(c.get("text", "")[:600] for c in chunks[:15])

    prompt = f"""
You are a database seeder. The business is: "{business_name}" ({business_type}).
Here is some context about the business from its documentation:
{context_text}

Return ONLY a JSON object with these keys:
{{
  "services": [
    {{"name": "...", "description": "...", "price": 0.00, "duration_min": 0, "category": "...", "modifiers": "size:small/medium/large;extras:cheese/bacon"}}
  ],
  "order_status_options": ["completed", "completed", "completed", "cancelled", "pending"]
}}

Rules:
- Extract ALL services/items mentioned in the context. Include every menu item, service, or product with its price when shown.
- If the context has a menu, table, or list of offerings, include each one. Do not omit items.
- Prices must match the context when provided.
- duration_min should reflect how long the service takes (e.g. cleaning=60, consultation=30).
- Include modifier options where applicable.
Respond with only the JSON, no markdown.
"""
    data = None
    for attempt in range(3):
        raw = llm_call(prompt, max_tokens=4000)  # large doc = many services
        raw = _re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            data = json.loads(raw)
            break
        except json.JSONDecodeError:
            if attempt == 2:
                raise ValueError(
                    f"LLM failed to return valid JSON for services extraction after 3 attempts. "
                    f"Business: {business_name}. Check document context or API."
                )
            continue

    if not data or not data.get("services"):
        raise ValueError(
            f"No services extracted from document for {business_name}. "
            "Ensure the document contains menu items, services, or products with prices."
        )

    conn = _get_db()
    try:
        # ── 1. Insert services ──────────────────────────────────────
        service_ids = []
        service_durations = {}
        for svc in data.get("services", []):
            cur = conn.execute(
                "INSERT INTO services (business_name, name, description, price, duration_min, category, modifiers) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    business_name,
                    svc.get("name", "Service"),
                    svc.get("description", ""),
                    float(svc.get("price", 50.0)),
                    int(svc.get("duration_min", 30)),
                    svc.get("category", "General"),
                    svc.get("modifiers", ""),
                ),
            )
            service_ids.append(cur.lastrowid)
            service_durations[cur.lastrowid] = int(svc.get("duration_min", 30))

        # ── 2. Insert named service providers ──────────────────────
        named_providers = _get_named_providers(business_type)
        provider_ids = []
        for p in named_providers:
            cur = conn.execute(
                "INSERT INTO service_providers (business_name, name, specialty, rating, available, schedule) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    business_name,
                    p["name"],
                    p["specialty"],
                    round(random.uniform(4.2, 5.0), 1),
                    1,  # all named providers are available
                    json.dumps(p["schedule"]),
                ),
            )
            provider_ids.append(cur.lastrowid)

        # Store provider name→id mapping for future-appointment seeding
        prov_name_to_id = {p["name"]: pid for p, pid in zip(named_providers, provider_ids)}

        # ── 3. Insert named + random clients ───────────────────────
        user_ids = []
        named_clients = _get_named_clients()
        named_client_ids = {}

        for c in named_clients:
            cur = conn.execute(
                "INSERT OR IGNORE INTO users "
                "(username, email, password_hash, full_name, phone, address, postal_code, city, family_members) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (c["username"], c["email"], c["password_hash"], c["full_name"],
                 c["phone"], c["address"], c["postal_code"], c["city"], c["family_members"]),
            )
            uid = cur.lastrowid
            if uid:
                user_ids.append(uid)
                named_client_ids[c["full_name"]] = uid

        families = [
            ([("spouse", "Female"), ("child", "Male")], "married_with_child"),
            ([("spouse", "Male")], "married"),
            ([], "single"),
            ([("child", "Female")], "parent"),
        ]
        pw_hash = bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode()
        for _ in range(12):
            family_info, _ = random.choice(families)
            family_json = json.dumps([
                {"relation": r, "gender": g, "name": fake.first_name_female() if g == "Female" else fake.first_name_male()}
                for r, g in family_info
            ]) if family_info else "[]"
            cur = conn.execute(
                "INSERT OR IGNORE INTO users "
                "(username, email, password_hash, full_name, phone, address, postal_code, city, family_members) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (fake.user_name(), fake.email(), pw_hash, fake.name(),
                 fake.phone_number(), fake.street_address(), fake.postcode(),
                 fake.city(), family_json),
            )
            if cur.lastrowid:
                user_ids.append(cur.lastrowid)

        # ── 4. Historical past orders (all clients) ─────────────────
        statuses = data.get("order_status_options", ["completed"] * 5)
        for uid in user_ids:
            for _ in range(random.randint(2, 6)):
                days_ago = random.randint(7, 365)
                scheduled = datetime.now() - timedelta(days=days_ago)
                svc_id = random.choice(service_ids)
                prov_id = random.choice(provider_ids)
                status = random.choice(statuses)
                svc_row = conn.execute("SELECT price, name FROM services WHERE id = ?", (svc_id,)).fetchone()
                price = svc_row["price"] if svc_row else 50.0
                conn.execute(
                    "INSERT INTO orders "
                    "(business_name, user_id, service_id, provider_id, status, order_type, scheduled_at, completed_at, "
                    "notes, total_price, delivery_type, items_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        business_name, uid, svc_id, prov_id, status,
                        random.choice(["order", "appointment"]),
                        scheduled.isoformat(),
                        (scheduled + timedelta(hours=1)).isoformat() if status == "completed" else None,
                        fake.sentence(nb_words=6),
                        price,
                        random.choice(["pickup", "delivery", "in-person"]),
                        json.dumps([{"name": svc_row["name"], "qty": 1, "price": price}]) if svc_row else "[]",
                    ),
                )

        # ── 5. Future confirmed appointments (conflict-test data) ───
        #
        # Strategy: scatter confirmed bookings across the next 14 days
        # so that:
        #   a) Several providers have fully-booked slots the agent must refuse.
        #   b) Overlapping-duration conflicts are clearly seeded:
        #      e.g. Ana gets 4:00-5:00 PM → Steven requesting 4:30 PM is blocked.
        #
        # We pick the first provider from named_providers as "Dr. Peter" equivalent.
        now = datetime.now()  # Use local time consistently with booking tools
        first_svc_id = service_ids[0] if service_ids else None
        first_prov_id = provider_ids[0] if provider_ids else None
        second_prov_id = provider_ids[1] if len(provider_ids) > 1 else first_prov_id

        if first_svc_id and first_prov_id and user_ids:
            # Determine duration for first service (default 60 min)
            dur1 = service_durations.get(first_svc_id, 60)

            # Named client user IDs (fallback to first random user)
            ana_id     = named_client_ids.get("Ana Thompson")     or user_ids[0]
            steven_id  = named_client_ids.get("Steven Williams")  or (user_ids[1] if len(user_ids) > 1 else user_ids[0])
            john_id    = named_client_ids.get("John Carter")      or (user_ids[2] if len(user_ids) > 2 else user_ids[0])
            hammad_id  = named_client_ids.get("Hammad Ali")       or (user_ids[3] if len(user_ids) > 3 else user_ids[0])
            robert_id  = named_client_ids.get("Robert Simpson")   or (user_ids[4] if len(user_ids) > 4 else user_ids[0])

            def _insert_future_appt(uid, svc_id, prov_id, start_dt, dur_min, label=""):
                """Insert a confirmed future appointment + calendar event."""
                end_dt = start_dt + timedelta(minutes=dur_min)
                svc_row = conn.execute("SELECT price, name FROM services WHERE id = ?", (svc_id,)).fetchone()
                price = svc_row["price"] if svc_row else 80.0
                svc_name = svc_row["name"] if svc_row else "Service"
                cur = conn.execute(
                    "INSERT INTO orders "
                    "(business_name, user_id, service_id, provider_id, status, order_type, scheduled_at, "
                    "total_price, notes, delivery_type, items_json) "
                    "VALUES (?, ?, ?, ?, 'confirmed', 'appointment', ?, ?, ?, 'in-person', ?)",
                    (
                        business_name, uid, svc_id, prov_id,
                        start_dt.isoformat(),
                        price,
                        label or f"Seeded future appointment for {svc_name}",
                        json.dumps([{"name": svc_name, "qty": 1, "price": price}]),
                    ),
                )
                order_id = cur.lastrowid
                conn.execute(
                    "INSERT INTO calendar_events "
                    "(business_name, user_id, order_id, title, description, start_time, end_time, provider, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'confirmed')",
                    (
                        business_name, uid, order_id,
                        f"Appointment: {svc_name}",
                        f"Booking #{order_id} for {svc_name}. Label: {label}",
                        start_dt.isoformat(),
                        end_dt.isoformat(),
                        named_providers[provider_ids.index(prov_id)]["name"] if prov_id in provider_ids else "Provider",
                    ),
                )
                return order_id

            # ── Scenario A: ANA blocks 4:00 PM - end (dur1) on day+3
            #    STEVEN requesting 4:30 PM on same day → should be blocked
            day3 = (now + timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
            # ANA: 4:00 PM to 4:00+dur1 min with first provider
            ana_start = day3.replace(hour=16, minute=0)
            _insert_future_appt(ana_id, first_svc_id, first_prov_id, ana_start, dur1,
                                 "Ana-conflict-seed: 4pm slot, blocks 4:30")

            # ── Scenario B: Several slots on day+5 fully booked for first provider
            day5 = (now + timedelta(days=5)).replace(hour=0, minute=0, second=0, microsecond=0)
            _insert_future_appt(john_id,   first_svc_id, first_prov_id, day5.replace(hour=9,  minute=0), dur1, "John-9am")
            _insert_future_appt(hammad_id, first_svc_id, first_prov_id, day5.replace(hour=10, minute=0), dur1, "Hammad-10am")
            _insert_future_appt(robert_id, first_svc_id, first_prov_id, day5.replace(hour=11, minute=0), dur1, "Robert-11am")
            _insert_future_appt(ana_id,    first_svc_id, first_prov_id, day5.replace(hour=14, minute=0), dur1, "Ana-2pm")

            # ── Scenario C: Second provider has a morning block on day+2
            day2 = (now + timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
            _insert_future_appt(steven_id, first_svc_id, second_prov_id, day2.replace(hour=10, minute=0), dur1, "Steven-10am-prov2")
            _insert_future_appt(john_id,   first_svc_id, second_prov_id, day2.replace(hour=14, minute=0), dur1, "John-2pm-prov2")

            # ── Scatter a few more random future bookings (days 1–14)
            all_named = [ana_id, steven_id, john_id, hammad_id, robert_id]
            for _ in range(8):
                future_day = (now + timedelta(days=random.randint(1, 14))).replace(
                    hour=random.choice([9, 10, 11, 13, 14, 15, 16]),
                    minute=0, second=0, microsecond=0)
                uid_ = random.choice(all_named)
                svc_ = random.choice(service_ids)
                prov_ = random.choice(provider_ids)
                dur_ = service_durations.get(svc_, 30)
                _insert_future_appt(uid_, svc_, prov_, future_day, dur_, "random-future")

        # ── 6. Loyalty points ───────────────────────────────────────
        for uid in user_ids:
            points = random.randint(0, 1500)
            tier = "gold" if points > 1000 else "silver" if points > 500 else "bronze"
            conn.execute(
                "INSERT OR IGNORE INTO loyalty_points (business_name, user_id, points, tier) VALUES (?, ?, ?, ?)",
                (business_name, uid, points, tier),
            )

        key = f"seeded_{business_name.lower().replace(' ', '_')}"
        conn.execute("INSERT OR REPLACE INTO business_meta (key, value) VALUES (?, '1')", (key,))
        conn.commit()
    finally:
        conn.close()


def repair_business_if_empty(business_name: str, business_type: str, chunks: list) -> bool:
    """
    If business has no services, run supplement + seed_providers. Returns True if repair was done.
    """
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM services WHERE business_name = ?", (business_name,)
        ).fetchone()
        if not row or row["n"] > 0 or not chunks:
            return False
        conn.close()
        conn = None
        supplement_services_from_chunks(business_name, business_type, chunks)
        seed_providers_fallback(business_name, business_type)
        return True
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def seed_providers_fallback(business_name: str, business_type: str) -> None:
    """
    When generate_synthetic_data fails (e.g. LLM JSON error), add providers so
    booking can still work. Uses services from supplement_services_from_chunks.
    """
    conn = _get_db()
    try:
        existing = conn.execute(
            "SELECT COUNT(*) FROM service_providers WHERE business_name = ?",
            (business_name,),
        ).fetchone()[0]
        if existing > 0:
            return
        named_providers = _get_named_providers(business_type)
        for p in named_providers:
            conn.execute(
                "INSERT INTO service_providers (business_name, name, specialty, rating, available, schedule) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    business_name,
                    p["name"],
                    p["specialty"],
                    round(random.uniform(4.2, 5.0), 1),
                    1,
                    json.dumps(p["schedule"]),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def supplement_services_from_chunks(business_name: str, business_type: str, chunks: list) -> None:
    """
    Add items/services from document chunks that are missing from the services table.
    Generic: extracts any products, menu items, or services with prices from the PDF.
    """
    if not chunks:
        return

    import re as _re
    conn = _get_db()
    try:
        existing = {r["name"].lower().strip() for r in conn.execute(
            "SELECT name FROM services WHERE business_name = ?", (business_name,)
        ).fetchall()}

        # Use chunks that likely contain priced items (tables, lists, or price patterns)
        context_text = "\n".join(c.get("text", "")[:400] for c in chunks[:12])

        prompt = f"""Extract all items, products, or services with prices from this business document.
Return ONLY a JSON array: [{{"name": "Item Name", "price": 12.99, "duration_min": 30}}, ...]

Document text:
{context_text[:3000]}

Rules: Extract every item that has a price. Use exact prices from the text. Include duration_min when shown (e.g. "45 min" -> 45, "30–60 min" -> 45). Default 30 if not shown. No markdown."""
        raw = llm_call(prompt, max_tokens=4000)
        raw = _re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end <= start:
                return
            items = json.loads(raw[start:end])
        except json.JSONDecodeError:
            return

        added = 0
        for it in items:
            if not isinstance(it, dict) or not it.get("name"):
                continue
            name = str(it["name"]).strip()
            if name.lower() in existing:
                continue
            price = float(it.get("price", 10.0))
            try:
                dur = int(it.get("duration_min", 30))
            except (TypeError, ValueError):
                dur = 30
            dur = max(1, dur)  # ensure positive
            conn.execute(
                "INSERT INTO services (business_name, name, description, price, duration_min, category, modifiers) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (business_name, name, "", price, dur, "General", ""),
            )
            existing.add(name.lower())
            added += 1
        if added:
            conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def get_global_stats() -> dict:
    conn = _get_db()
    try:
        stats = {}
        stats["total_customers"] = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        stats["total_orders"] = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        stats["total_revenue"] = conn.execute("SELECT COALESCE(SUM(total_price),0) FROM orders WHERE status='completed'").fetchone()[0]
        stats["avg_order_value"] = conn.execute("SELECT COALESCE(AVG(total_price),0) FROM orders WHERE status='completed'").fetchone()[0]
        stats["total_providers"] = conn.execute("SELECT COUNT(*) FROM service_providers").fetchone()[0]
        stats["total_services"] = conn.execute("SELECT COUNT(*) FROM services").fetchone()[0]
        stats["completed_orders"] = conn.execute("SELECT COUNT(*) FROM orders WHERE status='completed'").fetchone()[0]
        stats["cancelled_orders"] = conn.execute("SELECT COUNT(*) FROM orders WHERE status='cancelled'").fetchone()[0]

        # Most popular service
        popular = conn.execute(
            "SELECT s.name, COUNT(o.id) as cnt FROM orders o JOIN services s ON o.service_id=s.id GROUP BY s.name ORDER BY cnt DESC LIMIT 1"
        ).fetchone()
        stats["most_popular_service"] = popular["name"] if popular else "N/A"

        # Most popular provider
        top_prov = conn.execute(
            "SELECT sp.name, COUNT(o.id) as cnt FROM orders o JOIN service_providers sp ON o.provider_id=sp.id GROUP BY sp.name ORDER BY cnt DESC LIMIT 1"
        ).fetchone()
        stats["most_popular_provider"] = top_prov["name"] if top_prov else "N/A"

        return stats
    finally:
        conn.close()