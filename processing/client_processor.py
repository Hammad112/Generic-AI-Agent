"""
processing/client_processor.py
--------------------------------
Client data processor — handles INPUT and OUTPUT of client files.

INPUT  → Accept a JSON or CSV file of client records and seed them into the DB.
OUTPUT → Export all DB clients (with their history and fake slots) to a file.

Supported input formats:
  JSON: [{"full_name": "...", "email": "...", "phone": "...", "notes": "..."}, ...]
  CSV:  full_name,email,phone,notes (with header row)
  TXT:  One client per line: "Name | email | phone"

Usage (from main.py):
    from processing.client_processor import import_clients_file, export_clients_file
    import_clients_file("clients.json", business_name)
    export_clients_file(business_name, "output_clients.json")
"""

import os
import csv
import json
import sqlite3
import io
from pathlib import Path
from datetime import datetime, timezone
from faker import Faker

fake = Faker()


def _get_db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


# ─────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────

def _parse_json_clients(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):          # allow {clients: [...]} wrapper
            data = data.get("clients", data.get("data", [data]))
        return [dict(r) for r in data if isinstance(r, dict)]
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_csv_clients(raw: str) -> list[dict]:
    clients = []
    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        # Normalise common column name variations
        normalised = {}
        for k, v in row.items():
            key = (k or "").strip().lower().replace(" ", "_")
            normalised[key] = (v or "").strip()
        clients.append(normalised)
    return clients


def _parse_txt_clients(raw: str) -> list[dict]:
    """Parse lines like: John Smith | john@email.com | +1-555-1234"""
    clients = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in re.split(r"[|,\t]", line)]
        client: dict = {}
        if parts:
            client["full_name"] = parts[0]
        if len(parts) > 1:
            client["email"] = parts[1]
        if len(parts) > 2:
            client["phone"] = parts[2]
        if len(parts) > 3:
            client["notes"] = parts[3]
        if client.get("full_name"):
            clients.append(client)
    return clients


import re


def _normalise_client(raw: dict) -> dict:
    """Map common field name variations to canonical field names."""
    mapping = {
        "name":       "full_name",
        "client":     "full_name",
        "customer":   "full_name",
        "mail":       "email",
        "e_mail":     "email",
        "telephone":  "phone",
        "tel":        "phone",
        "mobile":     "phone",
        "address":    "address",
        "city":       "city",
        "postal":     "postal_code",
        "zip":        "postal_code",
        "postcode":   "postal_code",
        "note":       "notes",
        "comment":    "notes",
    }
    out = {}
    for k, v in raw.items():
        key = (k or "").strip().lower()
        out[mapping.get(key, key)] = v
    return out


# ─────────────────────────────────────────────
# Public: Import clients file → seed DB
# ─────────────────────────────────────────────

def import_clients_file(
    file_path: str,
    business_name: str,
    verbose: bool = True,
) -> dict:
    """
    Read a JSON / CSV / TXT file of client records and upsert them into the users table.

    Returns a summary: {"success": True, "inserted": N, "updated": N, "skipped": N}
    """
    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    raw = path.read_text(encoding="utf-8", errors="replace")
    ext = path.suffix.lower()

    if ext == ".json":
        records = _parse_json_clients(raw)
    elif ext == ".csv":
        records = _parse_csv_clients(raw)
    elif ext in (".txt", ".md"):
        records = _parse_txt_clients(raw)
    else:
        return {"success": False, "error": f"Unsupported format: {ext}. Use .json, .csv, or .txt"}

    if not records:
        return {"success": False, "error": "No client records found in file."}

    conn = _get_db()
    inserted = updated = skipped = 0
    try:
        for raw_rec in records:
            rec = _normalise_client(raw_rec)
            full_name = rec.get("full_name", "").strip()
            email     = rec.get("email", "").strip()

            if not full_name:
                skipped += 1
                continue

            # Build username
            base_uname = re.sub(r"[^a-z0-9]", "_", full_name.lower())
            if not email:
                email = f"{base_uname}_{fake.unique.random_int(100, 9999)}@example.com"

            # Check existing by email or full_name
            existing = conn.execute(
                "SELECT id FROM users WHERE email = ? OR full_name = ? LIMIT 1",
                (email, full_name),
            ).fetchone()

            if existing:
                # Update record
                conn.execute(
                    """UPDATE users SET
                           full_name    = COALESCE(?, full_name),
                           phone        = COALESCE(?, phone),
                           address      = COALESCE(?, address),
                           postal_code  = COALESCE(?, postal_code),
                           city         = COALESCE(?, city)
                       WHERE id = ?""",
                    (
                        full_name or None,
                        rec.get("phone") or None,
                        rec.get("address") or None,
                        rec.get("postal_code") or None,
                        rec.get("city") or None,
                        existing["id"],
                    ),
                )
                updated += 1
            else:
                # Insert new user
                # Ensure unique username
                uname  = base_uname
                suffix = 0
                while conn.execute("SELECT id FROM users WHERE username = ?", (uname,)).fetchone():
                    suffix += 1
                    uname = f"{base_uname}_{suffix}"

                conn.execute(
                    """INSERT INTO users
                       (username, email, password_hash, full_name, phone, address, postal_code, city)
                       VALUES (?, ?, '*', ?, ?, ?, ?, ?)""",
                    (
                        uname, email, full_name,
                        rec.get("phone", ""),
                        rec.get("address", ""),
                        rec.get("postal_code", ""),
                        rec.get("city", ""),
                    ),
                )
                inserted += 1

        conn.commit()

        if verbose:
            print(
                f"  [CLIENTS] Imported from {path.name}: "
                f"{inserted} new, {updated} updated, {skipped} skipped"
            )

        return {
            "success":  True,
            "inserted": inserted,
            "updated":  updated,
            "skipped":  skipped,
            "source":   str(path),
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Public: Export clients → file
# ─────────────────────────────────────────────

def export_clients_file(
    business_name: str,
    output_path: str = "clients_export.json",
    verbose: bool = True,
) -> dict:
    """
    Export all clients who have ever interacted with this business
    (i.e. have at least one order) to a JSON or CSV file.

    Includes: full_name, email, phone, city, total_orders, last_visit, loyalty_points
    """
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT u.id, u.full_name, u.email, u.phone, u.city,
                      COUNT(o.id)              as total_orders,
                      MAX(o.scheduled_at)      as last_visit,
                      COALESCE(SUM(o.total_price), 0) as total_spent
               FROM users u
               JOIN orders o ON o.user_id = u.id
               WHERE o.business_name = ?
               GROUP BY u.id
               ORDER BY last_visit DESC""",
            (business_name,),
        ).fetchall()
    finally:
        conn.close()

    clients = [dict(r) for r in rows]

    path = Path(output_path)
    ext  = path.suffix.lower()

    if ext == ".csv":
        if not clients:
            path.write_text("full_name,email,phone,city,total_orders,last_visit,total_spent\n")
        else:
            keys = ["full_name", "email", "phone", "city", "total_orders", "last_visit", "total_spent"]
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(clients)
    else:
        # Default: JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"business": business_name, "exported_at": datetime.now(timezone.utc).isoformat(),
                 "total": len(clients), "clients": clients},
                f, indent=2, default=str,
            )

    if verbose:
        print(f"  [CLIENTS] Exported {len(clients)} clients → {output_path}")

    return {"success": True, "total": len(clients), "output": output_path}


# ─────────────────────────────────────────────
# Public: Generate fake clients file
# ─────────────────────────────────────────────

def generate_fake_clients_file(
    count: int = 20,
    output_path: str = "fake_clients.json",
    verbose: bool = True,
) -> str:
    """
    Generate a fake clients file using Faker.
    This can be fed back into import_clients_file() as input.
    Returns the output file path.
    """
    clients = []
    for _ in range(count):
        profile = fake.simple_profile()
        clients.append({
            "full_name": profile["name"],
            "email":     profile["mail"],
            "phone":     fake.phone_number()[:20],
            "address":   fake.street_address(),
            "city":      fake.city(),
            "postal_code": fake.postcode(),
        })

    path = Path(output_path)
    ext  = path.suffix.lower()

    if ext == ".csv":
        keys = ["full_name", "email", "phone", "address", "city", "postal_code"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(clients)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clients, f, indent=2)

    if verbose:
        print(f"  [CLIENTS] Generated {count} fake clients → {output_path}")

    return output_path