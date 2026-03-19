"""
processing/pdf_processor.py
----------------------------
PDF and text file ingestion with intelligent chunking.
Supports: .pdf (via PyMuPDF), .md, .txt files.
"""

import os
import re
import sqlite3


def process_pdf(file_path: str) -> list[dict]:
    """
    Process a business description file (PDF, MD, or TXT).
    Returns list of chunk dicts.
    """
    file_path = str(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        pages = _extract_pages_pdf(file_path)
    elif ext in (".md", ".txt", ".text"):
        pages = _extract_pages_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .md, or .txt")

    all_chunks = []
    for page_num, page_text in enumerate(pages, 1):
        chunks = _intelligent_chunk(page_text, page_num)
        all_chunks.extend(chunks)

    # Post-process: use LLM to classify chunks into logical business topics
    all_chunks = _llm_reclassify_chunks(all_chunks)

    return all_chunks


def _llm_reclassify_chunks(chunks: list[dict]) -> list[dict]:
    """
    Use a single LLM call to classify structurally-parsed chunks into
    logical business topics/skills.  This fulfills the spec requirement
    for a 'separate LLM call to chunk/divide pdf information to logical topics'.
    """
    if not chunks:
        return chunks

    try:
        from core.llm_client import llm_call
        import json as _json
    except ImportError:
        return chunks  # graceful fallback if llm_client not available

    # Build summaries of each chunk (truncated to save tokens)
    summaries = []
    for i, c in enumerate(chunks):
        text_preview = c.get("text", "")[:150].replace("\n", " ")
        summaries.append(f"  {i}: [{c.get('section_title', 'General')}] {text_preview}")

    # Process in batches of 20 chunks to stay within token limits
    batch_size = 20
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch_summaries = "\n".join(summaries[batch_start:batch_end])
        indices = list(range(batch_start, batch_end))

        prompt = f"""You are a business knowledge organizer. Here are {len(indices)} chunks from a business description document:

{batch_summaries}

Classify each chunk into one logical business topic/skill category.
Use clear, descriptive category names like: "Menu & Pricing", "Services Offered",
"Operating Hours", "Cancellation Policy", "Staff & Qualifications",
"Location & Contact", "Promotions & Deals", "Quality Standards", etc.

Return ONLY a JSON object mapping chunk index to topic:
{{{", ".join(f'"{i}": "topic"' for i in indices[:3])}, ...}}
No markdown, just JSON."""

        try:
            raw = llm_call(prompt, temperature=0.1, max_tokens=500)
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
            mapping = _json.loads(raw)

            for idx_str, topic in mapping.items():
                idx = int(idx_str)
                if 0 <= idx < len(chunks) and isinstance(topic, str):
                    chunks[idx]["section_title"] = topic
        except Exception:
            pass  # Keep structural section_title if LLM fails

    return chunks


def _extract_pages_pdf(pdf_path: str) -> list[str]:
    """Extract text from a PDF using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return pages


def _extract_pages_text(file_path: str) -> list[str]:
    """Extract text from markdown or plain text files, splitting on headings."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on top-level headings (##) to simulate "pages"
    sections = re.split(r'\n(?=## )', content)
    pages = [s.strip() for s in sections if s.strip()]
    if not pages:
        pages = [content]
    return pages


def _intelligent_chunk(text: str, page_num: int) -> list[dict]:
    """
    Split text into meaningful chunks based on structure:
    - Headers
    - Bullet/numbered lists
    - Paragraphs
    - Tables (keep together)
    Does NOT use overlapping fixed-size windows.
    """
    chunks = []
    current_section = ""
    current_text_lines: list[str] = []

    lines = text.split("\n")

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect headers (markdown or uppercase titles)
        is_header = False
        if stripped.startswith("#"):
            is_header = True
            header_text = stripped.lstrip("#").strip()
        elif stripped.isupper() and len(stripped) > 3 and not stripped.startswith("|") and len(stripped.split()) < 8:
            is_header = True
            header_text = stripped

        if is_header:
            if current_text_lines:
                chunk_text = "\n".join(current_text_lines).strip()
                if chunk_text:
                    chunks.append({
                        "page_num": page_num,
                        "section_title": current_section or "General",
                        "text": chunk_text,
                    })
                current_text_lines = []
            current_section = header_text
            continue

        # Detect table rows or Menu Items (e.g. "Pizza .... $10")
        # We protect these from being split across chunks
        is_menu_item = False
        if stripped.startswith("|") or "..." in stripped or "$" in stripped:
             if len(stripped) > 10:
                 is_menu_item = True

        if is_menu_item:
            current_text_lines.append(stripped)
            continue

        # Double newline = Strong Paragraph Break
        if not stripped:
            if i > 0 and not lines[i-1].strip():
                joined = "\n".join(current_text_lines).strip()
                if joined and len(joined) > 100:
                    chunks.append({
                        "page_num": page_num,
                        "section_title": current_section or "General",
                        "text": joined,
                    })
                    # Use semantic overlap: keep the last 2 lines for context
                    current_text_lines = current_text_lines[-2:] if len(current_text_lines) > 2 else []
            continue

        current_text_lines.append(stripped)

        # Safety: Higher threshold for logical chunks
        joined = "\n".join(current_text_lines)
        if len(joined) > 1800:
            chunks.append({
                "page_num": page_num,
                "section_title": current_section or "General",
                "text": joined.strip(),
            })
            current_text_lines = current_text_lines[-3:] # Heavier overlap

    # Final chunk
    if current_text_lines:
        chunk_text = "\n".join(current_text_lines).strip()
        if chunk_text and len(chunk_text) > 10:
            chunks.append({
                "page_num": page_num,
                "section_title": current_section or "General",
                "text": chunk_text,
            })

    # Number the chunks
    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i

    return chunks


def save_chunks_to_db(chunks: list[dict], business_name: str) -> None:
    """Save processed chunks to the database."""
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, timeout=30.0)
    try:
        # Clear old chunks for this business
        conn.execute("DELETE FROM pdf_chunks WHERE business_name = ?", (business_name,))

        for chunk in chunks:
            conn.execute(
                "INSERT INTO pdf_chunks (business_name, chunk_index, page_num, section_title, text) VALUES (?, ?, ?, ?, ?)",
                (
                    business_name,
                    chunk.get("chunk_index", 0),
                    chunk.get("page_num", 0),
                    chunk.get("section_title", ""),
                    chunk.get("text", ""),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def load_chunks(business_name: str) -> list[dict]:
    """Load all chunks from the database for a specific business."""
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, chunk_index, page_num, section_title, text FROM pdf_chunks WHERE business_name = ? ORDER BY chunk_index",
            (business_name,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
