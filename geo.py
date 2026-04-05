"""
geo.py — Lightweight IP → country lookup.

Uses the ip-api.com free API (no key needed, 45 req/min limit).
Falls back to None if lookup fails — fraud engine handles None gracefully.

Usage:
    from geo import get_country
    country = get_country("203.0.113.42")  # returns "US" or None
"""

import httpx

GEO_API = "http://ip-api.com/json/{ip}?fields=countryCode,status"

# IPs that should never be looked up
PRIVATE_PREFIXES = ("127.", "192.168.", "10.", "172.16.", "::1", "localhost")


def get_country(ip: str) -> str | None:
    """
    Look up the country code for an IP address.
    Returns a 2-letter country code like "US", "RU", "CN"
    Returns None if the IP is private or the lookup fails.
    """
    if not ip:
        return None

    # Skip private/local IPs — common in development
    if any(ip.startswith(p) for p in PRIVATE_PREFIXES):
        return None

    try:
        response = httpx.get(
            GEO_API.format(ip=ip),
            timeout=3.0  # don't slow down login if geo is down
        )
        data = response.json()
        if data.get("status") == "success":
            return data.get("countryCode")
    except Exception:
        pass  # geo failure should never break login

    return None