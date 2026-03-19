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
    name        TEXT    NOT NULL,
    specialty   TEXT,
    rating      REAL    DEFAULT 4.5,
    available   INTEGER DEFAULT 1,
    schedule    TEXT
);

-- Services / products offered by the business
CREATE TABLE IF NOT EXISTS services (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
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
    user_id     INTEGER UNIQUE REFERENCES users(id),
    points      INTEGER DEFAULT 0,
    tier        TEXT    DEFAULT 'bronze',
    updated_at  TEXT    DEFAULT (datetime('now'))
);

-- PDF knowledge chunks
CREATE TABLE IF NOT EXISTS pdf_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
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
    topic       TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    source      TEXT    DEFAULT 'llm_enrichment',
    topic_tags  TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- Chat messages
CREATE TABLE IF NOT EXISTS conversations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
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

-- Customer complaints / dispute log
CREATE TABLE IF NOT EXISTS complaints (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER REFERENCES users(id),
    order_id            INTEGER REFERENCES orders(id),
    conversation_id     TEXT,
    complaint_type      TEXT,
    description         TEXT,
    suggested_resolution TEXT,
    status              TEXT    DEFAULT 'open',
    created_at          TEXT    DEFAULT (datetime('now'))
);

-- Conversation-specific cart state (NEW)
CREATE TABLE IF NOT EXISTS conversation_cart (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT UNIQUE NOT NULL,
    user_id     INTEGER REFERENCES users(id),
    cart_json   TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now'))
);