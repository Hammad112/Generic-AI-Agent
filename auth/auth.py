"""
auth/auth.py
------------
CLI-based authentication: login and registration.
Import path changed from: auth → auth.auth
"""

import sqlite3
import os
import bcrypt
from rich.console import Console
from rich.prompt import Prompt

console = Console()


def _get_db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def authenticate() -> dict:
    console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    console.print("[bold white]  Welcome! Please login or register.[/]")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]\n")

    while True:
        choice = Prompt.ask(
            "[bold yellow]Choose an option[/]",
            choices=["login", "register", "exit"],
            default="login",
        )
        if choice == "exit":
            raise SystemExit(0)
        elif choice == "login":
            user = _login()
            if user:
                return user
        else:
            user = _register()
            if user:
                return user


def _login() -> dict | None:
    console.print("\n[bold green]── Login ──[/]")
    username = Prompt.ask("[cyan]Username[/]")
    password = Prompt.ask("[cyan]Password[/]", password=True)

    conn = _get_db()
    try:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            console.print("[red]✗ User not found.[/]\n")
            return None
        if not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
            console.print("[red]✗ Incorrect password.[/]\n")
            return None
        console.print(f"\n[bold green]✓ Welcome back, {row['full_name'] or username}![/]\n")
        return dict(row)
    finally:
        conn.close()


def _register() -> dict | None:
    console.print("\n[bold green]── Register ──[/]")
    username = Prompt.ask("[cyan]Choose a username[/]")
    email = Prompt.ask("[cyan]Email address[/]")
    full_name = Prompt.ask("[cyan]Full name[/]")
    phone = Prompt.ask("[cyan]Phone number (optional)[/]", default="")
    password = Prompt.ask("[cyan]Choose a password[/]", password=True)
    confirm = Prompt.ask("[cyan]Confirm password[/]", password=True)

    if password != confirm:
        console.print("[red]✗ Passwords do not match.[/]\n")
        return None

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    conn = _get_db()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, full_name, phone) VALUES (?, ?, ?, ?, ?)",
            (username, email, pw_hash, full_name, phone),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        # Get current business name from database
        business_name = None
        try:
            business_name = conn.execute("SELECT value FROM business_meta WHERE key = 'business_name'").fetchone()
            if business_name:
                business_name = business_name["value"]
        except:
            business_name = None
        
        conn.execute(
            "INSERT OR IGNORE INTO loyalty_points (business_name, user_id, points, tier) VALUES (?, ?, ?, ?)",
            (business_name, cur.lastrowid, 100, "bronze"),
        )
        conn.commit()
        # console.print(f"\n[bold green]✓ Account created! Welcome, {full_name}! 🎉[/]")
        # console.print("[dim]You've been awarded 100 welcome loyalty points.[/]\n")
        return dict(row)
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            console.print("[red]✗ Username already taken.[/]\n")
        elif "email" in str(e):
            console.print("[red]✗ Email already registered.[/]\n")
        else:
            console.print(f"[red]✗ Registration failed: {e}[/]\n")
        return None
    finally:
        conn.close()
