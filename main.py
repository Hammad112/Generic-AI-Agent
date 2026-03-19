"""
main.py
-------
Entry point for the Generic AI Customer Service Agent.
Handles: PDF input, startup pipeline, authentication, conversation loop.

Usage:
    python main.py                        # interactive — will ask for PDF
    python main.py --pdf salon.pdf
    python main.py --pdf sample_pdfs/mario_pizza.md
"""

import argparse
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from dotenv import load_dotenv

load_dotenv()

from core.database import (
    init_db,
    generate_synthetic_data,
    repair_business_if_empty,
    seed_providers_fallback,
    supplement_services_from_chunks,
    set_business_meta,
    get_business_meta,
    list_businesses,
    business_is_loaded,
    migrate_add_business_name,
)
from processing.pdf_processor import process_pdf, load_chunks, save_chunks_to_db
from processing.knowledge_enricher import detect_business_type, enrich_knowledge, enrich_per_item
from auth.auth import authenticate
from agent.agent import run_agent_turn

console = Console()


# ── Startup Pipeline ──────────────────────────

def _process_new_business(file_path: str) -> tuple[str, str, list[dict]]:
    """
    Process a new business PDF/MD/TXT file and store it in the database.
    Returns (business_name, business_type, all_chunks).
    """
    with console.status("[bold green]Starting up..."):
        console.print(f"[cyan]Processing:[/] {file_path}")
        all_chunks = process_pdf(file_path)
        console.print(f"  → Created [green]{len(all_chunks)}[/] knowledge chunks")

        sample_text = "\n".join(c.get("text", "")[:200] for c in all_chunks[:5])

    with console.status("[bold]Detecting business type..."):
        business_info = detect_business_type(sample_text)

    business_name = business_info.get("business_name", "Business")
    business_type = business_info.get("business_type", "general")
    console.print(f"  → Detected: [bold]{business_name}[/] ({business_type})")

    # Save per-business metadata (scoped by business name)
    set_business_meta("business_type",  business_type,  business_name)
    set_business_meta("business_loaded", "1",            business_name)

    # Save chunks under this business name
    save_chunks_to_db(all_chunks, business_name)

    with console.status("[bold]Enriching knowledge base..."):
        enrich_knowledge(sample_text, business_type, business_name)
    console.print("  → Knowledge enrichment [green]complete[/]")

    with console.status("[bold]Supplementing menu from PDF..."):
        supplement_services_from_chunks(business_name, business_type, all_chunks)
    console.print("  → Menu items [green]complete[/]")

    with console.status("[bold]Generating synthetic data..."):
        try:
            generate_synthetic_data(business_type, business_name, all_chunks)
            console.print("  → Synthetic data [green]generated[/]")
        except ValueError:
            seed_providers_fallback(business_name, business_type)
            console.print(f"  → Synthetic data [yellow]partial[/] (LLM extraction failed, using supplement + providers)")

    with console.status("[bold]Enriching per-item knowledge..."):
        enrich_per_item(business_name, business_type)
    console.print("  → Per-item knowledge [green]complete[/]")

    # Store business hours if present in chunks
    hours_meta = get_business_meta("business_hours", business_name)
    if not hours_meta:
        for chunk in all_chunks:
            text_lower = chunk.get("text", "").lower()
            title_lower = chunk.get("section_title", "").lower()
            if "hour" in title_lower or ("hour" in text_lower and "open" in text_lower):
                set_business_meta("business_hours", chunk.get("text", "")[:500], business_name)
                break

    # Run migration to add business_name column to orders/calendar_events if needed
    migrate_add_business_name(business_name)

    return business_name, business_type, all_chunks


def startup(file_path: str | None = None) -> tuple[str, str, list[dict]]:
    """
    Multi-business startup pipeline.

    Behaviour:
      - If --pdf given and that business is NOT yet loaded → process + store it.
      - If --pdf given and that business IS already loaded → ask: use cached or re-process.
      - If no --pdf given and businesses exist → show picker.
      - If no businesses at all → prompt for a file.
    """
    # Always init DB first
    with console.status("[bold green]Initializing database..."):
        init_db()

    loaded_businesses = list_businesses()

    # ── Case 1: specific file provided ─────────────────────────────────────
    if file_path:
        # Quick-detect the business name from the file (cheap: first 5 chunks, no LLM)
        # by checking if any pdf_chunks rows already exist for this file path
        # We detect by processing first, then checking cache

        # Check by file stem as heuristic (user can always force reload with y)
        file_stem = Path(file_path).stem.replace("_", " ").replace("-", " ")

        # Word-level overlap matching: "mario_pizza" stem should match "Marios Pizza"
        def _names_overlap(stem: str, biz: str) -> bool:
            s, b = stem.lower(), biz.lower()
            if s in b or b in s:
                return True
            # Check if any word (4+ chars) from stem appears in biz name
            for word in s.split():
                if len(word) >= 4 and word in b:
                    return True
            return False

        match = next(
            (b for b in loaded_businesses if _names_overlap(file_stem, b["business_name"])),
            None,
        )

        if match:
            biz_name  = match["business_name"]
            biz_type  = get_business_meta("business_type", biz_name) or "general"
            console.print(
                f"[yellow]Business already loaded:[/] [bold]{biz_name}[/bold] "
                f"({match['chunk_count']} chunks, last loaded {match['last_loaded'][:10]})"
            )
            answer = Prompt.ask("Use cached data?", choices=["y", "n"], default="y")
            if answer == "y":
                with console.status("[bold green]Loading cached business data..."):
                    chunks = load_chunks(biz_name)
                return biz_name, biz_type, chunks
            else:
                # Force re-process
                return _process_new_business(file_path)
        else:
            # Brand new business
            return _process_new_business(file_path)

    # ── Case 2: no file given, businesses exist → show picker ───────────────
    if loaded_businesses:
        console.print()
        console.print(Panel(
            "[bold cyan]Loaded Businesses[/bold cyan]\n"
            "Each business is stored separately — pick one to chat with\n"
            "or load a new one from a PDF/MD/TXT file.",
            title="📋 Business Manager",
            border_style="cyan",
        ))

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#",             style="dim",  width=4)
        table.add_column("Business Name", style="bold", width=35)
        table.add_column("Type",          style="cyan", width=20)
        table.add_column("Chunks",        style="dim",  width=8)
        table.add_column("Last Loaded",   style="dim",  width=12)

        for i, biz in enumerate(loaded_businesses, 1):
            biz_type = get_business_meta("business_type", biz["business_name"]) or "general"
            table.add_row(
                str(i),
                biz["business_name"],
                biz_type,
                str(biz["chunk_count"]),
                biz["last_loaded"][:10] if biz["last_loaded"] else "—",
            )

        console.print(table)
        console.print(
            f"[dim]Enter a number (1–{len(loaded_businesses)}) to select, "
            f"or press Enter to load a new business file.[/dim]"
        )

        choice = Prompt.ask("Your choice", default="new").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(loaded_businesses):
            selected = loaded_businesses[int(choice) - 1]
            biz_name = selected["business_name"]
            biz_type = get_business_meta("business_type", biz_name) or "general"
            with console.status(f"[bold green]Loading {biz_name}..."):
                chunks = load_chunks(biz_name)
            console.print(f"  → Loaded [green]{len(chunks)}[/] chunks for [bold]{biz_name}[/]")
            # Repair: if business has no services (e.g. prior generate_synthetic_data failed), supplement now
            try:
                from core.database import repair_business_if_empty
                if repair_business_if_empty(biz_name, biz_type, chunks):
                    console.print("  → [green]Services and providers repaired[/]")
            except Exception:
                pass
            return biz_name, biz_type, chunks
        else:
            # Load new
            pdf_path = _prompt_for_pdf()
            return _process_new_business(str(pdf_path))

    # ── Case 3: no businesses, no file → prompt ─────────────────────────────
    pdf_path = _prompt_for_pdf()
    return _process_new_business(str(pdf_path))


# ── Interactive PDF prompt ────────────────────

def _prompt_for_pdf() -> Path:
    """Interactively ask the user for a PDF or business document path."""
    console.print(
        Panel(
            "[bold cyan]Welcome to the Generic AI Customer Service Agent[/]\n\n"
            "This agent can handle orders, appointments, and customer service\n"
            "for [bold]any business[/] — just provide a description file.\n\n"
            "Supported formats: [green].pdf[/], [green].md[/], [green].txt[/]",
            title="🤖 AI Agent",
            border_style="cyan",
        )
    )

    while True:
        choice = Prompt.ask(
            "Enter the path to your business description file (e.g., salon.pdf or info.md)"
        )

        path = Path(choice.strip())
        if path.exists() and path.is_file():
            return path
        
        if not path.exists():
            console.print(f"[red]Error:[/] File not found at [bold]{path}[/]. Please check the path and try again.")
        else:
            console.print(f"[red]Error:[/] [bold]{path}[/] is not a valid file. Please provide a path to a .pdf, .md, or .txt file.")


# ── Conversation loop ─────────────────────────

def conversation_loop(
    user: dict,
    business_name: str,
    business_type: str,
    all_chunks: list[dict],
) -> None:
    """Main interactive conversation loop."""
    conversation_id = str(uuid.uuid4())
    user_id = user["id"]

    console.print()
    console.rule(f"[bold green]Chat with {business_name} AI Agent[/]")
    console.print(
        "[dim]Type your message below. Type 'quit' or 'exit' to end.\n"
        "Type 'history' to see your order history.\n"
        "Type 'cart' to see your cart.\n"
        "Type 'clear' to start a new conversation.[/]\n"
    )

    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/]")
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()
        if cmd in ("quit", "exit", "bye"):
            console.print("[dim]Goodbye! 👋[/]")
            break
        elif cmd == "clear":
            conversation_id = str(uuid.uuid4())
            console.print("[yellow]New conversation started.[/]")
            continue

        # Run agent
        with console.status("[dim]Thinking...[/]"):
            try:
                agent_out = run_agent_turn(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    query=user_input,
                    all_chunks=all_chunks,
                    business_name=business_name,
                    business_type=business_type,
                )
                response_text = agent_out.get("response", "")
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                continue

        console.print(f"\n[bold green]Agent:[/] {response_text}\n")
        
        if "ics_file_url" in agent_out and agent_out["ics_file_url"]:
            console.print(f"[bold yellow]Calendar Event Generated:[/] {agent_out['ics_file_url']}\n")


# ── Main ──────────────────────────────────────

# ── Log Viewer ─────────────────────────────────
def show_logs(n: int = 50, filter_type: str = "") -> None:
    """Print the last N log entries from the database in a readable format."""
    import sqlite3, json as _json
    from datetime import timezone

    db_name = os.getenv("DB_NAME", "business_agent.db")
    if not Path(db_name).exists():
        console.print(f"[red]Database not found:[/] {db_name}")
        console.print("[dim]Run the agent first to create the database.[/]")
        return

    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row

    # ── Tool calls ──────────────────────────────
    if not filter_type or filter_type in ("tools", "tool"):
        console.rule("[bold cyan]Recent Tool Calls[/]")
        rows = conn.execute(
            "SELECT timestamp, tool_name, inputs, outputs FROM tool_calls ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
        if not rows:
            console.print("[dim]No tool calls logged yet.[/]")
        else:
            table = Table(show_header=True, header_style="bold magenta", box=None)
            table.add_column("Time",     style="dim",   width=19)
            table.add_column("Tool",     style="cyan",  width=25)
            table.add_column("Success",  style="green", width=7)
            table.add_column("Message / Error",          width=70)
            for r in reversed(rows):
                ts   = r["timestamp"][:19] if r["timestamp"] else ""
                tool = r["tool_name"] or ""
                try:
                    out = _json.loads(r["outputs"]) if r["outputs"] else {}
                except Exception:
                    out = {}
                success = str(out.get("success", "?"))
                msg     = out.get("message", out.get("error", ""))[:70]
                color   = "green" if success == "True" else "red" if success == "False" else "dim"
                table.add_row(ts, tool, f"[{color}]{success}[/{color}]", msg)
            console.print(table)
        console.print()

    # ── Routing decisions ───────────────────────
    if not filter_type or filter_type in ("routing", "route"):
        console.rule("[bold cyan]Recent Routing Decisions[/]")
        rows2 = conn.execute(
            "SELECT timestamp, details FROM agent_logs WHERE event_type = 'routing_decision' ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
        if not rows2:
            console.print("[dim]No routing decisions logged yet.[/]")
        else:
            table2 = Table(show_header=True, header_style="bold magenta", box=None)
            table2.add_column("Time",   style="dim",    width=19)
            table2.add_column("Query",  style="white",  width=45)
            table2.add_column("Tools",  style="cyan",   width=35)
            for r in reversed(rows2):
                ts = r["timestamp"][:19] if r["timestamp"] else ""
                try:
                    det = _json.loads(r["details"]) if r["details"] else {}
                except Exception:
                    det = {}
                query = det.get("query", "")[:45]
                tools = str(det.get("routed_tools", det.get("routed_tool", "")))
                table2.add_row(ts, query, tools)
            console.print(table2)
        console.print()

    # ── LLM calls ───────────────────────────────
    if not filter_type or filter_type in ("llm",):
        console.rule("[bold cyan]Recent LLM Calls[/]")
        rows3 = conn.execute(
            "SELECT timestamp, event_type, details FROM agent_logs WHERE event_type LIKE 'llm_call:%' ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
        if not rows3:
            console.print("[dim]No LLM calls logged yet.[/]")
        else:
            table3 = Table(show_header=True, header_style="bold magenta", box=None)
            table3.add_column("Time",   style="dim",  width=19)
            table3.add_column("Type",   style="cyan", width=20)
            table3.add_column("Prompt snippet",       width=50)
            table3.add_column("Response snippet",     width=50)
            for r in reversed(rows3):
                ts   = r["timestamp"][:19] if r["timestamp"] else ""
                call_type = r["event_type"].replace("llm_call:", "")
                try:
                    det = _json.loads(r["details"]) if r["details"] else {}
                except Exception:
                    det = {}
                prompt   = det.get("prompt",   "")[:50]
                response = det.get("response", "")[:50]
                table3.add_row(ts, call_type, prompt, response)
            console.print(table3)
        console.print()

    # ── DB write audit ───────────────────────────
    if not filter_type or filter_type in ("db", "writes"):
        console.rule("[bold cyan]Recent DB Writes[/]")
        rows4 = conn.execute(
            "SELECT timestamp, event_type, details FROM agent_logs WHERE event_type LIKE 'db_write:%' ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
        if not rows4:
            console.print("[dim]No DB write logs yet.[/]")
        else:
            table4 = Table(show_header=True, header_style="bold magenta", box=None)
            table4.add_column("Time",      style="dim",  width=19)
            table4.add_column("Operation", style="cyan", width=35)
            table4.add_column("Details",                 width=60)
            for r in reversed(rows4):
                ts        = r["timestamp"][:19] if r["timestamp"] else ""
                operation = r["event_type"].replace("db_write:", "")
                try:
                    det = _json.loads(r["details"]) if r["details"] else {}
                except Exception:
                    det = {}
                details = str(det)[:60]
                table4.add_row(ts, operation, details)
            console.print(table4)
        console.print()

    # ── Conversation messages ────────────────────
    if not filter_type or filter_type in ("chat", "conversation"):
        console.rule("[bold cyan]Recent Conversation Messages[/]")
        rows5 = conn.execute(
            "SELECT timestamp, role, content FROM conversations ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
        if not rows5:
            console.print("[dim]No conversation messages logged yet.[/]")
        else:
            for r in reversed(rows5):
                ts   = r["timestamp"][:19] if r["timestamp"] else ""
                role = r["role"].upper()
                content = r["content"] or ""
                color = "blue" if role == "USER" else "green"
                console.print(f"[dim]{ts}[/] [{color}]{role}:[/{color}] {content[:200]}")
        console.print()

    # ── Summary stats ────────────────────────────
    console.rule("[bold cyan]Database Summary[/]")
    for table_name in ("conversations", "tool_calls", "agent_logs", "orders", "cart", "users"):
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            console.print(f"  [dim]{table_name:20}[/] {count:>6} rows")
        except Exception:
            pass

    conn.close()
    console.print()
    console.print(f"[dim]Database: {Path(db_name).resolve()}[/]")



def main():
    parser = argparse.ArgumentParser(
        description="Generic AI Customer Service Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                # interactive — will ask for file
  python main.py --pdf sample_pdfs/mario_pizza.md
  python main.py --pdf sample_pdfs/bright_smile_dental.md
  python main.py --pdf path/to/your_business.pdf
        """,
    )
    parser.add_argument(
        "--pdf",
        required=False,
        type=str,
        default=None,
        help="Path to the business description file (.pdf, .md, .txt)",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        default=False,
        help="Show recent logs (tool calls, routing, LLM calls, DB writes)",
    )
    parser.add_argument(
        "--log-n",
        type=int,
        default=30,
        metavar="N",
        help="Number of log entries to show per category (default: 30)",
    )
    parser.add_argument(
        "--log-filter",
        type=str,
        default="",
        metavar="TYPE",
        help="Filter logs: tools | routing | llm | db | chat",
    )
    args = parser.parse_args()

    # ── Log viewer mode ─────────────────────────
    if args.logs:
        show_logs(n=args.log_n, filter_type=args.log_filter)
        sys.exit(0)

    # ── Determine file path (optional — startup() shows picker if omitted) ──
    pdf_path_str: str | None = None
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            console.print(f"[red]Error: File not found:[/] {pdf_path}")
            sys.exit(1)
        pdf_path_str = str(pdf_path)

    # Check API keys
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red bold]Error:[/] No API keys found.\n"
            "Open [cyan].env[/] and add your GEMINI_API_KEY or OPENAI_API_KEY."
        )
        sys.exit(1)

    try:
        # Run startup pipeline (handles multi-business selection internally)
        business_name, business_type, all_chunks = startup(pdf_path_str)

        # Authentication
        console.rule("[bold cyan]Authentication[/]")
        user = authenticate()

        # Start conversation
        conversation_loop(user, business_name, business_type, all_chunks)

    except KeyboardInterrupt:
        console.print("\n[dim]Session ended.[/]")
    except Exception as e:
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()