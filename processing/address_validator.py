"""
processing/address_validator.py
--------------------------------
Address extraction and validation using LLM + regex.

Supports Canadian and US addresses.
Canadian postal codes: A1A 1A1 format
US ZIP codes: 12345 or 12345-6789 format

Used by: agent/tools.py set_delivery_address()
"""

import re
import os
import json


def extract_and_validate_address(query: str) -> dict | None:
    """
    Extract and validate a delivery address from a customer message.

    Returns dict with keys:
        valid (bool), address (str), postal_code (str),
        city (str), province_or_state (str), country (str), message (str)

    Returns None on total failure.
    """
    from core.llm_client import llm_call

    # Build extraction prompt — strict JSON, no hallucination
    prompt = (
        "Extract the delivery address from this customer message.\n"
        "\n"
        f'Customer message: "{query}"\n'
        "\n"
        "Return ONLY valid JSON with these exact fields:\n"
        "{\n"
        '  "found": true,\n'
        '  "street_number": "34",\n'
        '  "street_name": "Front Street",\n'
        '  "unit": "",\n'
        '  "city": "Toronto",\n'
        '  "province_or_state": "ON",\n'
        '  "postal_code": "N4K 4L7",\n'
        '  "country": "Canada"\n'
        "}\n"
        "\n"
        "RULES:\n"
        "- Only extract information explicitly present in the message.\n"
        "- Do NOT invent street names, cities, or postal codes.\n"
        "- If no address found: {\"found\": false}\n"
        "- Normalize Canadian postal codes to format A1A 1A1 (with space).\n"
        "- Normalize US ZIP to 5 digits.\n"
        "- No markdown. No prose. Just JSON.\n"
    )

    raw = llm_call(prompt, temperature=0.0, max_tokens=300)
    raw = re.sub(r"```(?:json)?\s*", "", raw.strip()).strip()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {
            "valid": False,
            "message": "I couldn't parse that address. Please provide it as: street number, street name, city, postal code.",
        }

    if not data.get("found", False):
        return {
            "valid": False,
            "message": "I didn't find a delivery address in your message. Please provide your full address including postal code.",
        }

    # Extract fields
    street_num  = data.get("street_number", "").strip()
    street_name = data.get("street_name", "").strip()
    unit        = data.get("unit", "").strip()
    city        = data.get("city", "").strip()
    prov_state  = data.get("province_or_state", "").strip()
    postal      = data.get("postal_code", "").strip().upper()
    country     = data.get("country", "").strip()

    # Build full address string
    street = f"{street_num} {street_name}".strip()
    if unit:
        street = f"{street}, Unit {unit}"

    parts = [p for p in [street, city, prov_state, postal, country] if p]
    full_address = ", ".join(parts)

    # Validate postal code format
    issues = []
    is_canadian = "canada" in country.lower() or prov_state.upper() in (
        "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"
    )
    is_american = "united states" in country.lower() or "usa" in country.lower() or (
        len(prov_state) == 2 and not is_canadian
    )

    if is_canadian and postal:
        # Remove space for validation then re-add
        pc_clean = postal.replace(" ", "")
        if not re.match(r"^[A-Z]\d[A-Z]\d[A-Z]\d$", pc_clean):
            issues.append(
                f"'{postal}' doesn't look like a valid Canadian postal code (should be A1A 1A1). "
                "Please double-check."
            )
        else:
            # Normalize: ensure space in middle
            postal = f"{pc_clean[:3]} {pc_clean[3:]}"
    elif is_american and postal:
        pc_clean = postal.replace("-", "").replace(" ", "")
        if not re.match(r"^\d{5}(\d{4})?$", pc_clean):
            issues.append(
                f"'{postal}' doesn't look like a valid US ZIP code (should be 12345 or 12345-6789)."
            )

    if not street:
        issues.append("Street address is missing.")
    if not city:
        issues.append("City is missing.")
    if not postal:
        issues.append("Postal/ZIP code is missing.")

    if issues:
        return {
            "valid": False,
            "address": full_address,
            "postal_code": postal,
            "city": city,
            "province_or_state": prov_state,
            "country": country,
            "issues": issues,
            "message": "There's an issue with the address: " + " ".join(issues) + " Please correct it.",
        }

    return {
        "valid": True,
        "address": full_address,
        "street": street,
        "city": city,
        "province_or_state": prov_state,
        "postal_code": postal,
        "country": country,
        "message": (
            f"Address confirmed:\n"
            f"  {street}\n"
            f"  {city}, {prov_state}  {postal}\n"
            f"  {country}"
        ),
    }