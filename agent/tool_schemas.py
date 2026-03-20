"""
agent/tool_schemas.py
----------------------
All tool function schemas in a format compatible with both
OpenAI function calling and Gemini tool use.

_OPENAI_TOOLS  → list[dict]  for openai client.chat.completions.create(tools=...)
_GEMINI_TOOLS  → list[dict]  for genai generate_content(tools=...)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Master schema list — single source of truth
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMAS = [
    # ── Greeting ─────────────────────────────────────────────────────────────
    {
        "name": "greet_customer",
        "description": (
            "Greet the customer warmly and show the business services and hours. "
            "Call when the customer says hello, hi, hey, good morning, or any greeting."
        ),
        "parameters": {},
    },

    # ── Cart / Ordering ───────────────────────────────────────────────────────
    {
        "name": "add_to_cart",
        "description": (
            "Add one or more items to the customer's cart. "
            "Call when the customer wants to order, buy, or add an item. "
            "Also when they say 'I want X', 'give me X', 'yes' after an agent offer."
        ),
        "parameters": {
            "item_name": {
                "type": "string",
                "description": "Name of the item(s) to add, comma-separated if multiple",
            },
            "quantity": {
                "type": "integer",
                "description": "Quantity to add (default 1)",
            },
            "modifiers": {
                "type": "string",
                "description": "Any customisations, e.g. 'extra cheese, large'",
            },
        },
    },
    {
        "name": "remove_from_cart",
        "description": (
            "Remove an item from the customer's cart. "
            "Call when the customer says 'remove X', 'I don't want X', 'cancel X', 'delete X'."
        ),
        "parameters": {
            "item_name": {
                "type": "string",
                "description": "Name of the item to remove",
            },
        },
    },
    {
        "name": "view_cart",
        "description": (
            "Show the customer's current cart contents and subtotal. "
            "Call when asked 'what's in my cart', 'show my order', 'what have I ordered'."
        ),
        "parameters": {},
    },
    {
        "name": "confirm_order",
        "description": (
            "Place / confirm the customer's current cart as a final order. "
            "Call when the customer says 'confirm', 'place order', 'that's all', "
            "'yes' after a final order summary, or 'checkout'."
        ),
        "parameters": {
            "notes": {
                "type": "string",
                "description": "Any special instructions for the order",
            },
        },
    },
    {
        "name": "set_delivery_type",
        "description": (
            "Set whether the order is for delivery or pickup. "
            "Call when the customer mentions delivery, takeaway, pickup, or collection."
        ),
        "parameters": {
            "delivery_type": {
                "type": "string",
                "enum": ["delivery", "pickup"],
                "description": "Either 'delivery' or 'pickup'",
            },
        },
    },
    {
        "name": "set_delivery_address",
        "description": (
            "Save the customer's delivery address. "
            "Call when the customer provides a street address or postal code for delivery."
        ),
        "parameters": {
            "address": {
                "type": "string",
                "description": "Full street address including postal code",
            },
        },
    },

    # ── Appointments ──────────────────────────────────────────────────────────
    {
        "name": "check_availability",
        "description": (
            "Check available appointment slots and providers. "
            "Call when the customer asks about availability, open slots, "
            "'when can I come in', or 'what times are free'."
        ),
        "parameters": {
            "date": {
                "type": "string",
                "description": (
                    "Requested date as a day name or date string, "
                    "e.g. 'monday', 'tomorrow', '2026-03-25'. Leave empty for general availability."
                ),
            },
            "service_name": {
                "type": "string",
                "description": "Service the customer wants (optional)",
            },
            "provider_name": {
                "type": "string",
                "description": "Preferred provider/doctor name (optional)",
            },
        },
    },
    {
        "name": "book_appointment",
        "description": (
            "Book an appointment for the customer. "
            "ONLY call when the customer has explicitly provided BOTH a date AND a time. "
            "Do NOT call if they only named a service without date+time — ask for those first."
        ),
        "parameters": {
            "service_name": {
                "type": "string",
                "description": "Name of the service to book",
            },
            "date": {
                "type": "string",
                "description": "Appointment date, e.g. 'monday', '2026-03-25', 'tomorrow'",
            },
            "time": {
                "type": "string",
                "description": "Appointment time, e.g. '10am', '14:00', '2pm'",
            },
            "provider_name": {
                "type": "string",
                "description": "Preferred provider name (optional)",
            },
        },
        "required": ["service_name", "date", "time"],
    },
    {
        "name": "reschedule_booking",
        "description": (
            "Reschedule an existing appointment to a new date/time. "
            "Call when the customer wants to change, move, or reschedule a booking."
        ),
        "parameters": {
            "new_date": {
                "type": "string",
                "description": "New appointment date",
            },
            "new_time": {
                "type": "string",
                "description": "New appointment time",
            },
        },
    },
    {
        "name": "cancel_booking",
        "description": (
            "Cancel an existing appointment or booking. "
            "Call when the customer wants to cancel or delete an appointment."
        ),
        "parameters": {
            "booking_ref": {
                "type": "string",
                "description": "Booking reference number (optional — agent can look up by user)",
            },
        },
    },

    # ── Information ───────────────────────────────────────────────────────────
    {
        "name": "get_pricing",
        "description": (
            "Get service/product prices filtered to the active topic. "
            "Call when the customer asks about prices, costs, or rates."
        ),
        "parameters": {
            "category_filter": {
                "type": "string",
                "description": (
                    "Optional category or keyword to filter services, "
                    "e.g. 'teeth', 'massage', 'pizza'. Leave empty for all."
                ),
            },
        },
    },
    {
        "name": "get_recommendations",
        "description": (
            "Give personalised recommendations based on customer history and popular items. "
            "Call when the customer asks what to try, what's popular, or for suggestions."
        ),
        "parameters": {
            "category_filter": {
                "type": "string",
                "description": "Optional filter to scope recommendations, e.g. 'hair', 'pizza'",
            },
        },
    },
    {
        "name": "get_business_info",
        "description": (
            "Get general business info: location, hours, staff, services overview. "
            "Call when asked 'tell me about you', 'where are you', 'who works here'."
        ),
        "parameters": {},
    },
    {
        "name": "get_business_hours",
        "description": (
            "Get opening hours and holiday schedule. "
            "Call when the customer asks what time you open/close, or holiday hours."
        ),
        "parameters": {},
    },
    {
        "name": "get_provider_info",
        "description": (
            "Get info about a specific staff member or provider. "
            "Call when the customer asks about a doctor, therapist, or staff member by name."
        ),
        "parameters": {
            "provider_name": {
                "type": "string",
                "description": "Name of the provider to look up (partial name ok)",
            },
        },
    },
    {
        "name": "search_knowledge",
        "description": (
            "Search the business knowledge base for any specific information: "
            "policies, FAQs, procedures, ingredients, aftercare, etc. "
            "Call when the customer asks a specific question not covered by other tools."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "The specific question or topic to search for",
            },
        },
    },
    {
        "name": "get_order_history",
        "description": (
            "Show the customer's past orders and bookings. "
            "Call when asked 'my previous orders', 'booking history', 'what have I ordered'."
        ),
        "parameters": {},
    },
    {
        "name": "get_faqs",
        "description": (
            "Retrieve frequently asked questions from the knowledge base. "
            "Call when the customer asks a general 'how does X work' question."
        ),
        "parameters": {},
    },
    {
        "name": "get_promotions",
        "description": (
            "Get current deals, promotions, and special offers. "
            "Call when the customer asks about discounts, deals, or offers."
        ),
        "parameters": {},
    },

    # ── Loyalty ───────────────────────────────────────────────────────────────
    {
        "name": "get_loyalty_balance",
        "description": (
            "Check the customer's loyalty points, tier, and rewards. "
            "Call when asked about points, rewards, or membership status."
        ),
        "parameters": {},
    },
    {
        "name": "apply_loyalty_discount",
        "description": (
            "Apply loyalty points as a discount. "
            "Call when the customer wants to redeem or use their loyalty points."
        ),
        "parameters": {},
    },

    # ── Customer Profile ──────────────────────────────────────────────────────
    {
        "name": "update_customer_profile",
        "description": (
            "Update the customer's contact details or preferences. "
            "Call when the customer wants to change their name, phone, email, or address."
        ),
        "parameters": {
            "field": {
                "type": "string",
                "description": "Field to update: 'phone', 'email', 'address', 'name'",
            },
            "value": {
                "type": "string",
                "description": "New value for the field",
            },
        },
    },
    {
        "name": "check_family_members",
        "description": (
            "Look up family members on the account for cross-selling. "
            "Call when recommending services for family or asking about family."
        ),
        "parameters": {},
    },

    # ── Calendar ──────────────────────────────────────────────────────────────
    {
        "name": "get_calendar",
        "description": (
            "Show the customer's upcoming appointments and schedule. "
            "Call when asked 'my appointments', 'my schedule', 'what's booked'."
        ),
        "parameters": {},
    },

    # ── Dispute ───────────────────────────────────────────────────────────────
    {
        "name": "handle_dispute",
        "description": (
            "Handle complaints, wrong orders, missing items, or quality issues. "
            "Call when the customer has a complaint or says something was wrong."
        ),
        "parameters": {
            "complaint": {
                "type": "string",
                "description": "Description of the complaint or issue",
            },
        },
    },

    # ── Escalation ────────────────────────────────────────────────────────────
    {
        "name": "escalate_to_human",
        "description": (
            "Escalate to a human agent. "
            "Call when the customer requests a human, is very frustrated, "
            "or when the AI cannot resolve the issue."
        ),
        "parameters": {
            "reason": {
                "type": "string",
                "description": "Reason for escalation",
            },
        },
    },

    # ── Web Search ────────────────────────────────────────────────────────────
    {
        "name": "search_web",
        "description": (
            "Search the web for current information not in the knowledge base. "
            "Call when the customer asks for news, trends, or external information."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Build provider-specific formats
# ─────────────────────────────────────────────────────────────────────────────

def _build_openai_tools() -> list[dict]:
    tools = []
    for s in _SCHEMAS:
        params = s.get("parameters", {})
        required = s.get("required", [])
        props = {}
        for k, v in params.items():
            prop = {"type": v.get("type", "string"), "description": v.get("description", "")}
            if "enum" in v:
                prop["enum"] = v["enum"]
            props[k] = prop
        tools.append({
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        })
    return tools


def _build_gemini_tools() -> list[dict]:
    """Build Gemini-compatible function declarations."""
    declarations = []
    for s in _SCHEMAS:
        params = s.get("parameters", {})
        props = {}
        for k, v in params.items():
            prop = {"type": v.get("type", "STRING").upper(), "description": v.get("description", "")}
            if "enum" in v:
                prop["enum"] = v["enum"]
            props[k] = prop
        decl = {
            "name": s["name"],
            "description": s["description"],
        }
        if props:
            decl["parameters"] = {
                "type": "OBJECT",
                "properties": props,
            }
            if s.get("required"):
                decl["parameters"]["required"] = s["required"]
        declarations.append(decl)
    return declarations


OPENAI_TOOLS = _build_openai_tools()
GEMINI_TOOL_DECLARATIONS = _build_gemini_tools()

# Quick name→schema lookup
TOOL_SCHEMA_BY_NAME = {s["name"]: s for s in _SCHEMAS}