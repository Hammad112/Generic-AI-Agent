"""
processing/calendar_processor.py
----------------------------------
Calendar file processor — handles both INPUT and OUTPUT of .ics / .txt files.

INPUT  → Accept an existing .ics or .txt file with time slot templates;
          parse those slots and seed them into the DB as available provider slots.
OUTPUT → Write all bookings/fake-slots back to .ics files in the calendar/ folder.

Usage (from main.py):
    from processing.calendar_processor import import_calendar_file, export_all_slots
    import_calendar_file("my_slots.ics", business_name, business_type)
    export_all_slots(business_name)
"""

import os
import re
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from faker import Faker

fake = Faker()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALENDAR_DIR   = os.path.join(_PROJECT_ROOT, "calendar")


def _get_db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


# ─────────────────────────────────────────────
# ICS parser helpers
# ─────────────────────────────────────────────

def _parse_ics_datetime(value: str) -> datetime | None:
    """Parse iCalendar DTSTART/DTEND values like 20260325T090000Z."""
    value = value.split(";")[-1]  # strip TZID=... param
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S", "%Y%m%d"):
        try:
            dt = datetime.strptime(value.strip(), fmt)
            if fmt.endswith("Z"):
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _parse_ics_blocks(text: str) -> list[dict]:
    """Extract VEVENT blocks from raw ICS text and return structured dicts."""
    events = []
    for block in re.split(r"BEGIN:VEVENT", text, flags=re.IGNORECASE):
        if "END:VEVENT" not in block.upper():
            continue
        # Unfold RFC 5545 folded lines (continuation with whitespace)
        block = re.sub(r"\r?\n[ \t]", "", block)

        def _prop(name: str) -> str:
            m = re.search(rf"^{name}[;:][^\r\n]*", block, re.MULTILINE | re.IGNORECASE)
            if not m:
                return ""
            val = m.group(0).split(":", 1)[-1].strip()
            return val

        dtstart = _parse_ics_datetime(_prop("DTSTART"))
        dtend   = _parse_ics_datetime(_prop("DTEND"))
        summary = _prop("SUMMARY")
        status  = _prop("STATUS") or "CONFIRMED"
        uid     = _prop("UID")
        attendee_raw = _prop("ATTENDEE")
        attendee_cn  = re.search(r"CN=([^;:]+)", attendee_raw, re.IGNORECASE)
        patient_name = attendee_cn.group(1).strip() if attendee_cn else ""

        if dtstart:
            events.append({
                "uid":           uid,
                "summary":       summary,
                "status":        status.upper(),
                "start":         dtstart,
                "end":           dtend or (dtstart + timedelta(hours=1)),
                "patient_name":  patient_name,
            })
    return events


def _parse_txt_slots(text: str) -> list[dict]:
    """
    Parse a plain-text slot file.
    Accepts lines like:
        2026-03-25 09:00  - Haircut (60 min)
        2026-03-25 10:30  - Massage (90 min)
    or ISO datetimes like 2026-03-25T09:00:00
    """
    slots = []
    # Match date-time patterns and optional label
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::\d{2})?(?:\s*[-–]\s*(.+?)(?:\((\d+)\s*min\))?)?$",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        date_part  = m.group(1)
        time_part  = m.group(2)
        label      = (m.group(3) or "Available Slot").strip()
        dur_min    = int(m.group(4)) if m.group(4) else 60

        try:
            start = datetime.fromisoformat(f"{date_part}T{time_part}:00")
        except ValueError:
            continue

        end = start + timedelta(minutes=dur_min)
        slots.append({
            "uid":          f"txt-slot-{date_part}-{time_part.replace(':', '')}",
            "summary":      label,
            "status":       "CONFIRMED",
            "start":        start,
            "end":          end,
            "patient_name": "",
        })
    return slots


# ─────────────────────────────────────────────
# Public: Import calendar file → seed DB
# ─────────────────────────────────────────────

def import_calendar_file(
    file_path: str,
    business_name: str,
    business_type: str = "",
    verbose: bool = True,
) -> dict:
    """
    Read an .ics or .txt file and seed the slots into the DB.

    For each parsed slot:
      - If it has a patient name → insert as an order + calendar_event for that client.
      - If it has no patient name → insert as a "blocked" slot on the first provider
        (makes that time unavailable for booking).

    Returns a summary dict.
    """
    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    raw = path.read_text(encoding="utf-8", errors="replace")
    ext = path.suffix.lower()

    if ext == ".ics":
        events = _parse_ics_blocks(raw)
    elif ext in (".txt", ".md"):
        events = _parse_txt_slots(raw)
    else:
        return {"success": False, "error": f"Unsupported file type: {ext}"}

    if not events:
        return {"success": False, "error": "No valid calendar events found in file."}

    conn = _get_db()
    try:
        # Find first available provider for this business
        provider = conn.execute(
            "SELECT id, name FROM service_providers WHERE business_name = ? LIMIT 1",
            (business_name,),
        ).fetchone()
        provider_id   = provider["id"]   if provider else None
        provider_name = provider["name"] if provider else "Staff"

        # Find a generic service to attach
        service = conn.execute(
            "SELECT id, name, duration_min, price FROM services WHERE business_name = ? LIMIT 1",
            (business_name,),
        ).fetchone()
        service_id   = service["id"]           if service else None
        service_name = service["name"]         if service else "Appointment"
        duration_min = service["duration_min"] if service else 60
        price        = service["price"]        if service else 0.0

        # Find or create a placeholder user for imported slots
        placeholder = conn.execute(
            "SELECT id FROM users WHERE username = 'calendar_import'",
        ).fetchone()
        if placeholder:
            placeholder_id = placeholder["id"]
        else:
            conn.execute(
                """INSERT INTO users (username, email, password_hash, full_name)
                   VALUES ('calendar_import','import@calendar.local','*','Calendar Import')""",
            )
            conn.commit()
            placeholder_id = conn.execute(
                "SELECT id FROM users WHERE username = 'calendar_import'"
            ).fetchone()["id"]

        inserted = 0
        skipped  = 0
        ics_files = []

        for ev in events:
            # Skip cancelled / declined events
            if ev["status"] in ("CANCELLED", "DECLINED"):
                # Generate a cancel .ics so the client sees the cancellation
                fname = _write_single_ics(ev, provider_name, business_name, method="CANCEL")
                ics_files.append(fname)
                skipped += 1
                continue

            # Check for conflict
            start_iso = ev["start"].isoformat()
            end_iso   = ev["end"].isoformat()
            conflict = conn.execute(
                """SELECT id FROM orders
                   WHERE provider_id = ?
                     AND status IN ('pending','confirmed')
                     AND datetime(scheduled_at) < datetime(?)
                     AND datetime(scheduled_at, '+' || COALESCE(
                         (SELECT duration_min FROM services WHERE id = orders.service_id), 30
                       ) || ' minutes') > datetime(?)
                   LIMIT 1""",
                (provider_id, end_iso, start_iso),
            ).fetchone()
            if conflict:
                skipped += 1
                continue

            # Resolve or create user for named client
            patient_name = ev["patient_name"] or fake.name()
            user_row = conn.execute(
                "SELECT id FROM users WHERE full_name = ?", (patient_name,)
            ).fetchone()
            if user_row:
                user_id = user_row["id"]
            else:
                uname = patient_name.lower().replace(" ", "_") + f"_{fake.unique.random_int(100,999)}"
                conn.execute(
                    """INSERT INTO users (username, email, password_hash, full_name)
                       VALUES (?, ?, '*', ?)""",
                    (uname, f"{uname}@example.com", patient_name),
                )
                conn.commit()
                user_id = conn.execute(
                    "SELECT id FROM users WHERE username = ?", (uname,)
                ).fetchone()["id"]

            # Insert order
            conn.execute(
                """INSERT INTO orders
                   (business_name, user_id, service_id, provider_id, status, order_type,
                    scheduled_at, total_price, notes)
                   VALUES (?, ?, ?, ?, 'confirmed', 'appointment', ?, ?, ?)""",
                (business_name, user_id, service_id, provider_id,
                 start_iso, price, f"Imported from {path.name}"),
            )
            conn.commit()
            order_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Insert calendar_event
            conn.execute(
                """INSERT INTO calendar_events
                   (business_name, user_id, order_id, title, description,
                    start_time, end_time, provider, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'confirmed')""",
                (business_name, user_id, order_id,
                 ev["summary"] or service_name,
                 f"Imported: {ev['uid']}",
                 start_iso, end_iso, provider_name),
            )
            conn.commit()

            # Write .ics output file for this slot
            fname = _write_single_ics(
                ev, provider_name, business_name, method="REQUEST",
                order_id=order_id, patient_name=patient_name,
            )
            ics_files.append(fname)
            inserted += 1

        if verbose:
            print(f"  [CALENDAR] Imported {inserted} slots, skipped {skipped} from {path.name}")

        return {
            "success": True,
            "inserted": inserted,
            "skipped":  skipped,
            "ics_files": ics_files,
            "source": str(path),
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Public: Export all DB slots → ICS files
# ─────────────────────────────────────────────

def export_all_slots(business_name: str, verbose: bool = True) -> list[str]:
    """
    Export every confirmed/pending calendar_event for this business
    to individual .ics files in the calendar/ folder.

    Returns list of written file paths.
    """
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT e.id, e.order_id, e.title, e.start_time, e.end_time,
                      e.provider, e.status, u.full_name as patient_name,
                      o.total_price
               FROM calendar_events e
               JOIN orders o ON e.order_id = o.id
               JOIN users  u ON e.user_id  = u.id
               WHERE e.business_name = ?
                 AND e.status != 'cancelled'
               ORDER BY e.start_time""",
            (business_name,),
        ).fetchall()
    finally:
        conn.close()

    os.makedirs(CALENDAR_DIR, exist_ok=True)
    written = []
    for r in rows:
        try:
            start = datetime.fromisoformat(r["start_time"])
            end   = datetime.fromisoformat(r["end_time"])
        except (TypeError, ValueError):
            continue

        ev = {
            "uid":          f"export-{r['id']}",
            "summary":      r["title"],
            "status":       (r["status"] or "confirmed").upper(),
            "start":        start,
            "end":          end,
            "patient_name": r["patient_name"] or "Customer",
        }
        fname = _write_single_ics(
            ev,
            provider_name=r["provider"] or "Staff",
            business_name=business_name,
            method="REQUEST",
            order_id=r["order_id"],
            patient_name=r["patient_name"] or "Customer",
        )
        written.append(fname)

    if verbose:
        print(f"  [CALENDAR] Exported {len(written)} slots to {CALENDAR_DIR}/")
    return written


# ─────────────────────────────────────────────
# Public: Generate fake slot ICS file
# ─────────────────────────────────────────────

def generate_fake_slots_file(
    business_name: str,
    days_ahead: int = 14,
    slots_per_day: int = 4,
    output_filename: str = "fake_slots.ics",
) -> str:
    """
    Generate a fake-slots .ics file using Faker-generated data.
    This can then be fed BACK into import_calendar_file() as input.

    Creates calendar/<output_filename> and returns the full path.
    """
    os.makedirs(CALENDAR_DIR, exist_ok=True)
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Generic AI Agent//Fake Slot Generator//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:REQUEST",
    ]

    now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    uid_counter = 1

    for day_offset in range(1, days_ahead + 1):
        base_day = now + timedelta(days=day_offset)
        if base_day.weekday() >= 5:          # skip weekends
            continue
        hour = 9
        for _ in range(slots_per_day):
            if hour >= 17:
                break
            start_dt = base_day.replace(hour=hour, minute=0)
            end_dt   = start_dt + timedelta(hours=1)
            name     = fake.name()
            dtstamp  = now.strftime("%Y%m%dT%H%M%SZ")
            dtstart  = start_dt.strftime("%Y%m%dT%H%M%SZ")
            dtend    = end_dt.strftime("%Y%m%dT%H%M%SZ")

            lines += [
                "BEGIN:VEVENT",
                f"UID:fakeslot-{uid_counter:04d}@genericagent",
                f"DTSTAMP:{dtstamp}",
                f"DTSTART:{dtstart}",
                f"DTEND:{dtend}",
                f"SUMMARY:Appointment – {business_name}",
                f"ATTENDEE;CN={name}:mailto:{name.lower().replace(' ','.')}@example.com",
                "STATUS:CONFIRMED",
                "END:VEVENT",
            ]
            uid_counter += 1
            hour += 1 + (uid_counter % 2)   # vary gap: 1 or 2 hours

    lines.append("END:VCALENDAR")

    filepath = os.path.join(CALENDAR_DIR, output_filename)
    with open(filepath, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\r\n".join(lines) + "\r\n")

    return filepath


# ─────────────────────────────────────────────
# Internal: write one .ics file
# ─────────────────────────────────────────────

def _write_single_ics(
    ev: dict,
    provider_name: str,
    business_name: str,
    method: str = "REQUEST",
    order_id: int = 0,
    patient_name: str = "",
) -> str:
    """Write one VEVENT to a .ics file and return its filename."""
    os.makedirs(CALENDAR_DIR, exist_ok=True)
    safe_name   = re.sub(r"[^a-z0-9_]", "_", (patient_name or "slot").lower())
    safe_method = method.upper()

    if safe_method == "CANCEL":
        filename = f"{safe_name}_cancel_{order_id or ev['uid']}.ics"
    else:
        filename = f"{safe_name}_appointment_{order_id or ev['uid']}.ics"

    filepath = os.path.join(CALENDAR_DIR, filename)
    start    = ev["start"]
    end      = ev["end"]

    try:
        start_utc = start.astimezone(timezone.utc)
        end_utc   = end.astimezone(timezone.utc)
    except (TypeError, AttributeError):
        start_utc = start.replace(tzinfo=timezone.utc)
        end_utc   = end.replace(tzinfo=timezone.utc)

    dtstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dtstart = start_utc.strftime("%Y%m%dT%H%M%SZ")
    dtend   = end_utc.strftime("%Y%m%dT%H%M%SZ")
    status  = "CANCELLED" if safe_method == "CANCEL" else "CONFIRMED"
    summary = ev.get("summary") or "Appointment"
    if safe_method == "CANCEL":
        summary = f"CANCELLED: {summary}"

    uid = ev.get("uid") or f"booking-{order_id}@genericagent"

    content = (
        f"BEGIN:VCALENDAR\r\n"
        f"VERSION:2.0\r\n"
        f"PRODID:-//Generic AI Agent//EN\r\n"
        f"CALSCALE:GREGORIAN\r\n"
        f"METHOD:{safe_method}\r\n"
        f"BEGIN:VEVENT\r\n"
        f"UID:{uid}\r\n"
        f"DTSTAMP:{dtstamp}\r\n"
        f"DTSTART:{dtstart}\r\n"
        f"DTEND:{dtend}\r\n"
        f"SUMMARY:{summary}\r\n"
        f"ORGANIZER;CN={business_name}:mailto:noreply@genericagent.com\r\n"
        f"ATTENDEE;CN={patient_name or 'Customer'}:mailto:customer@example.com\r\n"
        f"LOCATION:{business_name}\r\n"
        f"STATUS:{status}\r\n"
        f"SEQUENCE:0\r\n"
        f"END:VEVENT\r\n"
        f"END:VCALENDAR\r\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filename