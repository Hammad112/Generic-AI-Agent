"""
processing/knowledge_enricher.py
---------------------------------
Uses LLM to:
  1) detect_business_type() from PDF text
  2) enrich_knowledge() — generate supplementary knowledge not in the PDF
     (called "skills" in the spec): FAQs, product comparisons, tips, etc.
  3) enrich_with_web_search() — use OpenAI web search tool to pull LIVE knowledge
     about this business type (health warnings, trends, alternatives, etc.)
     e.g. "Pizza is not healthy → agent knows to proactively suggest healthy options"
"""

import os
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from core.llm_client import llm_call


def _get_db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def enrichment_already_done(business_name: str) -> bool:
    conn = _get_db()
    try:
        count = conn.execute("SELECT COUNT(*) FROM enriched_knowledge WHERE business_name = ?", (business_name,)).fetchone()[0]
        return count > 0
    finally:
        conn.close()


def detect_business_type(pdf_summary: str) -> dict:
    """
    Detect business name and type from PDF content.
    Returns dict: {"business_name": "...", "business_type": "..."}
    """
    prompt = f"""
Analyse this business document excerpt and identify the business:

---
{pdf_summary[:2000]}
---

Return ONLY a JSON object like:
{{
  "business_name": "Glow Beauty Salon",
  "business_type": "beauty salon"
}}

No markdown, just JSON.
"""
    raw = llm_call(prompt, temperature=0.1)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
        return {
            "business_name": data.get("business_name", "The Business"),
            "business_type": data.get("business_type", "general business"),
        }
    except json.JSONDecodeError:
        return {"business_name": "The Business", "business_type": "general business"}


def enrich_knowledge(pdf_text: str, business_type: str, business_name: str = "") -> list[dict]:
    """
    Generate supplementary "skills" — knowledge topics not in the PDF.
    Uses LLM to identify gaps and then write detailed articles for each.
    
    Args:
        pdf_text: Sample text from the ingested PDF for context
        business_type: e.g. "restaurant", "dental clinic"
        business_name: Optional name of the business
    """
    if not business_name:
        business_name = "the business"

    if enrichment_already_done(business_name):
        return load_enriched_from_db(business_name)

    topics_prompt = f"""
You are a strategic business analyst for "{business_name}" ({business_type}).

Here is the existing knowledge from their PDF:
---
{pdf_text[:3000]}
---

Identify 8-10 high-value knowledge topics ("skills") to add to this knowledge base.
Include at least THREE analytical "Business Skills" — these are observations about customer behavior and cross-sell opportunities (e.g., "Customers ordering pizza usually want a cold beverage" or "Patients coming for cleanings are often interested in whitening").

Focus on:
- Analytical Business Skills (proactive upselling insights)
- Frequently Asked Questions
- Detailed service comparisons
- Booking/Cancellation policies
- Staff expertise & unique selling points
- Seasonal promotion structures

Return ONLY a JSON array of topic names:
["Topic 1", "Topic 2", ...]
No markdown, just JSON.
"""
    raw = llm_call(topics_prompt, temperature=0.4)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        topics = json.loads(raw)
        if not isinstance(topics, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        topics = [
            "Frequently Asked Questions",
            "Pricing & Packages",
            "Booking & Cancellation Policy",
            "Loyalty Rewards Program",
            "Staff Qualifications",
            "Product/Service Comparisons",
            "Seasonal Promotions",
            "Tips & Best Practices",
        ]

    enriched = []
    
    # Use parallel execution for faster enrichment
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_enrich_topic, topic, business_name, business_type): topic for topic in topics}
        
        for future in futures:
            topic = futures[future]
            try:
                content = future.result()
                if content:
                    enriched.append({
                        "topic": topic,
                        "content": content,
                        "source": "llm_enrichment",
                        "topic_tags": topic.lower().replace(" ", ","),
                    })
            except Exception as e:
                print(f"Error enriching topic {topic}: {e}")

    _save_enriched(enriched, business_name)
    return enriched


def _enrich_topic(topic: str, business_name: str, business_type: str) -> str:
    """Generate a detailed knowledge article on the given topic."""
    prompt = (
        f'You are a knowledge base writer for "{business_name}" ({business_type}).\n'
        "\n"
        f'Write a helpful knowledge base article about: "{topic}"\n'
        "\n"
        "This content will be used by an AI customer service agent to answer customer questions.\n"
        "\n"
        "RULES:\n"
        "- Base content on industry-standard practices for this business type.\n"
        "- Do NOT invent specific prices, staff names, or policies not grounded in common industry knowledge.\n"
        "- If giving examples of prices or durations, clearly label them as 'typical' or 'industry average'.\n"
        "- Do NOT fabricate facts. If uncertain about a detail, omit it.\n"
        "- Write in clear paragraphs. Third person. Approximately 200-300 words.\n"
        "- Make it practical and specific to the business type.\n"
    )
    try:
        return llm_call(prompt, temperature=0.5, max_tokens=600)
    except Exception:
        return f"Information about {topic} for {business_name}."


def _save_enriched(enriched: list[dict], business_name: str) -> None:
    conn = _get_db()
    try:
        for item in enriched:
            conn.execute(
                "INSERT INTO enriched_knowledge (business_name, topic, content, source, topic_tags) VALUES (?, ?, ?, ?, ?)",
                (
                    business_name,
                    item["topic"],
                    item["content"],
                    item.get("source", "llm_enrichment"),
                    item.get("topic_tags"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo. Supports both package names:
      pip install ddgs              (new name)
      pip install duckduckgo-search (old name, now shows warning)
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise RuntimeError(
                "DuckDuckGo search not installed. Run: pip install ddgs"
            )
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as exc:
        raise RuntimeError(f"DuckDuckGo search failed: {exc}")


def enrich_with_web_search(business_name: str, business_type: str, services: list[str] = None) -> list[dict]:
    """
    Use DuckDuckGo (free, no API key) to pull live, factual knowledge about
    this business type and its services, then use the project's own LLM to
    synthesise that into knowledge-base articles.

    Flow:
      1. LLM generates 5-7 targeted search queries for this business type.
      2. DuckDuckGo fetches real web snippets for each query (free, instant).
      3. LLM synthesises the snippets into a 150-200 word knowledge article.
      4. Articles saved to enriched_knowledge table (source='web_search').

    Examples of knowledge added:
      - "Pizza is calorie-dense → suggest thin crust / veggie options"
      - "Teeth whitening sensitivity → advise avoiding cold food after"
      - "Massage contraindicated post-surgery → ask patient first"

    Falls back to LLM-only generation if DuckDuckGo is unavailable.
    Requires: pip install duckduckgo-search
    """
    if enrichment_web_done(business_name):
        return load_enriched_from_db(business_name, source_filter="web_search")

    # ── Step 1: LLM generates targeted search queries ────────────────────────
    service_snippet = ", ".join((services or [])[:10]) if services else ""
    topics_prompt = f"""You are a consumer-awareness advisor for "{business_name}" ({business_type}).

Identify 5-6 important real-world facts or warnings a customer service AI MUST know
to give responsible, helpful advice about: {business_type}.
{f'Services offered include: {service_snippet}.' if service_snippet else ''}

Examples of the KIND of knowledge needed:
- "Pizza calorie content healthy alternatives thin crust"
- "dental whitening sensitivity aftercare tips"
- "massage contraindications post surgery precautions"
- "dry cleaning chemical solvents clothing care advice"

Return ONLY a JSON array of short search query strings (5-8 words each):
["query 1", "query 2", ...]
No markdown, just JSON."""

    raw = llm_call(topics_prompt, temperature=0.3)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        search_queries = json.loads(raw)
        if not isinstance(search_queries, list):
            raise ValueError
        search_queries = [str(q) for q in search_queries[:6]]
    except (json.JSONDecodeError, ValueError):
        search_queries = [
            f"{business_type} health tips for customers",
            f"{business_type} common concerns and alternatives",
            f"{business_type} safety precautions advice",
        ]

    enriched = []
    ddg_available = True

    # ── Step 2 & 3: DDG fetch → LLM synthesise ──────────────────────────────
    for query in search_queries:
        raw_snippets = []

        # Try DuckDuckGo first
        if ddg_available:
            try:
                results = _ddg_search(query, max_results=4)
                raw_snippets = [
                    f"[{r.get('title', '')}] {r.get('body', '')}"
                    for r in results if r.get("body")
                ]
                if raw_snippets:
                    print(f"  [DDG] ✓ '{query}' → {len(raw_snippets)} results")
                    # Store search details so they appear in logs/all_logs.log
                    try:
                        from core.master_log import log_web_search_detail
                        log_web_search_detail(business_name, query, results)
                    except Exception:
                        pass
            except RuntimeError as e:
                err = str(e)
                if "not installed" in err:
                    ddg_available = False
                    print(f"  [DDG] duckduckgo-search not installed — using LLM fallback.")
                else:
                    print(f"  [DDG] ✗ '{query}' — {e}")

        # Build synthesis prompt (with real snippets if available, else LLM-only)
        if raw_snippets:
            snippets_block = "\n\n".join(raw_snippets[:4])
            synthesis_prompt = (
                f'You are a knowledge-base writer for "{business_name}" ({business_type}).\n\n'
                f'Using these real web search results for "{query}":\n\n'
                f"{snippets_block}\n\n"
                f"Write a concise 150-200 word knowledge-base article that a customer service AI "
                f"can use to give practical, accurate advice to customers. "
                f"Focus on what customers should KNOW and what the AI should RECOMMEND. "
                f"Do not just repeat the snippets — synthesise them into clear guidance."
            )
            source_tag = "web_search"
        else:
            # Pure LLM fallback — no DDG results
            synthesis_prompt = (
                f'You are a knowledge-base writer for "{business_name}" ({business_type}).\n\n'
                f'Write a 150-200 word article answering: "{query}"\n\n'
                f"Include practical advice a customer service AI should give. "
                f"Use general industry knowledge. Label estimates as 'typically' or 'generally'."
            )
            source_tag = "web_search_llm_fallback"

        try:
            content = llm_call(synthesis_prompt, temperature=0.4, max_tokens=500)
            if content.strip():
                enriched.append({
                    "topic":      f"[Web Knowledge] {query.title()}",
                    "content":    content.strip(),
                    "source":     source_tag,
                    "topic_tags": f"web,{source_tag},{business_type.lower().replace(' ', ',')}",
                })
        except Exception as exc:
            print(f"  [WEB SEARCH] LLM synthesis failed for '{query}': {exc}")

    if enriched:
        _save_enriched(enriched, business_name)
        ddg_count = sum(1 for e in enriched if e["source"] == "web_search")
        llm_count = len(enriched) - ddg_count
        print(
            f"  [WEB SEARCH] Added {len(enriched)} articles for {business_name} "
            f"({ddg_count} from DDG, {llm_count} LLM-only)"
        )

    return enriched


def enrichment_web_done(business_name: str) -> bool:
    """Check if web search enrichment has already been run for this business."""
    conn = _get_db()
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM enriched_knowledge WHERE business_name = ? AND source LIKE 'web_search%'",
            (business_name,),
        ).fetchone()[0]
        return count > 0
    finally:
        conn.close()


def load_enriched_from_db(business_name: str, source_filter: str = "") -> list[dict]:
    conn = _get_db()
    try:
        if source_filter:
            rows = conn.execute(
                "SELECT * FROM enriched_knowledge WHERE business_name = ? AND source LIKE ? ORDER BY id",
                (business_name, f"{source_filter}%"),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM enriched_knowledge WHERE business_name = ? ORDER BY id",
                (business_name,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def enrich_per_item(business_name: str, business_type: str) -> list[dict]:
    """
    Generate detailed knowledge for each individual service/product.
    Covers specs, pros/cons, who it's best for, comparisons — 
    fulfilling the spec requirement for deep item-level research.
    """
    conn = _get_db()
    try:
        services = conn.execute(
            "SELECT id, name, description, price, category FROM services WHERE business_name = ? ORDER BY id",
            (business_name,),
        ).fetchall()
    finally:
        conn.close()

    if not services:
        return []

    # Check if item-level enrichment already exists
    conn = _get_db()
    try:
        existing = conn.execute(
            "SELECT COUNT(*) FROM enriched_knowledge WHERE business_name = ? AND topic LIKE '[Item Detail]%'",
            (business_name,),
        ).fetchone()[0]
    finally:
        conn.close()

    if existing >= len(services):
        return []  # Already enriched

    enriched_items = []

    def _research_item(svc: dict) -> dict | None:
        prompt = f"""You are a product/service researcher for "{business_name}" ({business_type}).

Research and write detailed information about this specific item:
  Name: {svc['name']}
  Category: {svc['category'] or 'General'}
  Description: {svc['description'] or 'No description'}
  Price: ${svc['price']:.2f}

Include:
- Detailed description and what it involves
- Key features, ingredients, or specifications
- Pros and cons
- Who it's best suited for (ideal customer profile)
- How it compares to similar offerings in the {business_type} industry
- Common questions customers ask about this item
- Complementary items/services that pair well with it

Write 150-250 words. Be specific and factual.
"""
        try:
            content = llm_call(prompt, temperature=0.5, max_tokens=500)
            return {
                "topic": f"[Item Detail] {svc['name']}",
                "content": content,
                "source": "llm_item_enrichment",
                "topic_tags": f"item,{svc['category'] or 'general'},{svc['name'].lower().replace(' ', ',')}",
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_research_item, dict(s)): dict(s) for s in services}
        for future in futures:
            try:
                result = future.result()
                if result:
                    enriched_items.append(result)
            except Exception:
                pass

    if enriched_items:
        _save_enriched(enriched_items, business_name)

    return enriched_items