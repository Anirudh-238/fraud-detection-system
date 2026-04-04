"""
rule_engine.py — Layer 1: Rule-based fraud detection engine.
Fast, deterministic, zero-ML. Returns flags + a suggested action.

Each module function returns:
    (flags: List[str], action: RiskAction)

Thresholds are constants at the top — easy to tune without touching logic.
"""

from typing import List, Tuple
from datetime import datetime, timezone
from schemas import (
    LoginRequest, SignupRequest, PaymentRequest, PromoRequest, RiskAction
)

# ─────────────────────────────────────────────
# Tunable thresholds
# ─────────────────────────────────────────────

# Account Takeover
MAX_FAILED_ATTEMPTS_WARN  = 3
MAX_FAILED_ATTEMPTS_BLOCK = 8
HIGH_RISK_LOGIN_HOURS     = range(1, 5)     # 1 AM – 4 AM (local)

# Identity Theft
MAX_ACCOUNTS_PER_IP_WARN  = 3
MAX_ACCOUNTS_PER_IP_BLOCK = 6
DISPOSABLE_EMAIL_DOMAINS  = {
    "mailinator.com", "tempmail.xyz", "throwaway.email",
    "guerrillamail.com", "yopmail.com", "sharklasers.com",
    "trashmail.com", "maildrop.cc", "dispostable.com",
    "fakeinbox.com", "spamgourmet.com", "spam4.me",
}

# Payment Fraud
HIGH_VALUE_THRESHOLD_USD  = 500.0
VERY_HIGH_VALUE_USD       = 2000.0
MAX_CARDS_24H_WARN        = 2
MAX_CARDS_24H_BLOCK       = 4
MAX_ORDERS_24H_WARN       = 5
MAX_ORDERS_24H_BLOCK      = 10
NEW_ACCOUNT_DAYS          = 7       # account age < this is suspicious

# Promotion Abuse
MAX_ACCOUNTS_IP_PROMO     = 3
MAX_ACCOUNTS_DEVICE_PROMO = 3
MAX_REDEMPTIONS_WARN      = 1
MAX_REDEMPTIONS_BLOCK     = 3


# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def _parse_hour(timestamp_str: str) -> int:
    """Extract hour-of-day from an ISO-8601 string. Returns -1 on failure."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.hour
    except (ValueError, TypeError):
        return -1


def _is_disposable_email(email: str) -> bool:
    domain = email.split("@")[-1].lower()
    return domain in DISPOSABLE_EMAIL_DOMAINS


def _country_distance_risk(country_a: str, country_b: str) -> bool:
    """
    Very lightweight impossible-travel heuristic.
    Returns True if the two countries are considered high-risk pairs.
    A real system would use IP geolocation + time delta.
    """
    if not country_a or not country_b:
        return False
    # Treat different countries logged within same session as suspicious
    return country_a.upper() != country_b.upper()


# ─────────────────────────────────────────────
# Module 1 — Account Takeover
# ─────────────────────────────────────────────

def check_account_takeover(req: LoginRequest) -> Tuple[List[str], RiskAction]:
    """
    Rules:
    1. Too many failed attempts → block
    2. Country change between sessions → flag
    3. New device + foreign country → flag
    4. Login at suspicious hour (1–4 AM) + new device → flag
    5. IP changed + new device → flag
    """
    flags: List[str] = []
    severity = 0  # 0=clean, 1=warn, 2=block

    # Rule 1 — Brute-force / credential stuffing
    if req.failed_attempts >= MAX_FAILED_ATTEMPTS_BLOCK:
        flags.append(f"BRUTE_FORCE: {req.failed_attempts} failed attempts (threshold={MAX_FAILED_ATTEMPTS_BLOCK})")
        severity = max(severity, 2)
    elif req.failed_attempts >= MAX_FAILED_ATTEMPTS_WARN:
        flags.append(f"MANY_FAILED_LOGINS: {req.failed_attempts} failed attempts")
        severity = max(severity, 1)

    # Rule 2 — Impossible travel (country change)
    if _country_distance_risk(req.last_known_country, req.current_country):
        flags.append(
            f"IMPOSSIBLE_TRAVEL: country changed from "
            f"'{req.last_known_country}' to '{req.current_country}'"
        )
        severity = max(severity, 2)

    # Rule 3 — New device from foreign country
    if req.new_device and req.current_country and req.last_known_country:
        if req.current_country.upper() != req.last_known_country.upper():
            flags.append("NEW_DEVICE_FOREIGN_COUNTRY: new device used from a different country")
            severity = max(severity, 2)

    # Rule 4 — Suspicious login hour
    hour = _parse_hour(req.login_timestamp)
    if hour in HIGH_RISK_LOGIN_HOURS:
        flags.append(f"ODD_HOUR_LOGIN: login at {hour:02d}:xx local time")
        severity = max(severity, 1)
        if req.new_device:
            flags.append("ODD_HOUR_NEW_DEVICE: odd-hour login on a new device")
            severity = max(severity, 2)

    # Rule 5 — IP change on new device
    if req.new_device and req.last_known_ip and req.ip_address != req.last_known_ip:
        flags.append(
            f"IP_CHANGE_NEW_DEVICE: IP changed from {req.last_known_ip} to {req.ip_address}"
        )
        severity = max(severity, 1)

    action = _severity_to_action(severity)
    return flags, action


# ─────────────────────────────────────────────
# Module 2 — Identity Theft (Signup)
# ─────────────────────────────────────────────

def check_identity_theft(req: SignupRequest) -> Tuple[List[str], RiskAction]:
    """
    Rules:
    1. SSN reuse → block
    2. Email reuse → block
    3. Name / DOB mismatch → flag
    4. Too many accounts from same IP → flag / block
    5. Disposable email domain → flag
    """
    flags: List[str] = []
    severity = 0

    # Rule 1 — Duplicate SSN (synthetic identity / account factory)
    if req.ssn_seen_before:
        flags.append("DUPLICATE_SSN: SSN (last 4) already registered in system")
        severity = max(severity, 2)

    # Rule 2 — Email already registered
    if req.email_seen_before:
        flags.append(f"DUPLICATE_EMAIL: {req.email} already registered")
        severity = max(severity, 2)

    # Rule 3 — Name / DOB mismatch (synthetic ID signal)
    if req.name_dob_mismatch:
        flags.append("NAME_DOB_MISMATCH: provided name does not match expected DOB pattern")
        severity = max(severity, 1)

    # Rule 4 — Account farm from single IP
    if req.accounts_from_ip >= MAX_ACCOUNTS_PER_IP_BLOCK:
        flags.append(
            f"IP_ACCOUNT_FARM: {req.accounts_from_ip} accounts from IP {req.ip_address}"
        )
        severity = max(severity, 2)
    elif req.accounts_from_ip >= MAX_ACCOUNTS_PER_IP_WARN:
        flags.append(
            f"MANY_ACCOUNTS_IP: {req.accounts_from_ip} accounts from IP {req.ip_address}"
        )
        severity = max(severity, 1)

    # Rule 5 — Disposable email
    if _is_disposable_email(req.email):
        flags.append(f"DISPOSABLE_EMAIL: domain '{req.email.split('@')[-1]}' is flagged")
        severity = max(severity, 1)

    action = _severity_to_action(severity)
    return flags, action


# ─────────────────────────────────────────────
# Module 3 — Payment Fraud (Checkout)
# ─────────────────────────────────────────────

def check_payment_fraud(req: PaymentRequest) -> Tuple[List[str], RiskAction]:
    """
    Rules:
    1. Very high transaction value → block
    2. High value on new account → block
    3. Multiple cards in 24h → flag / block
    4. High order velocity → flag / block
    5. Billing / shipping country mismatch → flag
    6. Address mismatch (billing ≠ shipping address) → flag
    7. Digital goods + high value + new account → block
    """
    flags: List[str] = []
    severity = 0

    # Rule 1 — Extreme transaction value
    if req.amount_usd >= VERY_HIGH_VALUE_USD:
        flags.append(f"VERY_HIGH_VALUE: ${req.amount_usd:.2f} exceeds ${VERY_HIGH_VALUE_USD:.2f}")
        severity = max(severity, 2)
    elif req.amount_usd >= HIGH_VALUE_THRESHOLD_USD:
        flags.append(f"HIGH_VALUE: ${req.amount_usd:.2f} exceeds ${HIGH_VALUE_THRESHOLD_USD:.2f}")
        severity = max(severity, 1)

    # Rule 2 — High value + brand-new account
    if req.amount_usd >= HIGH_VALUE_THRESHOLD_USD and req.account_age_days < NEW_ACCOUNT_DAYS:
        flags.append(
            f"HIGH_VALUE_NEW_ACCOUNT: ${req.amount_usd:.2f} on account only "
            f"{req.account_age_days} day(s) old"
        )
        severity = max(severity, 2)

    # Rule 3 — Card velocity
    if req.cards_used_24h >= MAX_CARDS_24H_BLOCK:
        flags.append(f"CARD_VELOCITY: {req.cards_used_24h} different cards used in 24h")
        severity = max(severity, 2)
    elif req.cards_used_24h >= MAX_CARDS_24H_WARN:
        flags.append(f"MULTIPLE_CARDS: {req.cards_used_24h} cards used in 24h")
        severity = max(severity, 1)

    # Rule 4 — Order velocity
    if req.orders_24h >= MAX_ORDERS_24H_BLOCK:
        flags.append(f"ORDER_VELOCITY: {req.orders_24h} orders in 24h")
        severity = max(severity, 2)
    elif req.orders_24h >= MAX_ORDERS_24H_WARN:
        flags.append(f"HIGH_ORDER_COUNT: {req.orders_24h} orders in 24h")
        severity = max(severity, 1)

    # Rule 5 — Cross-country billing / shipping
    if req.billing_country.upper() != req.shipping_country.upper():
        flags.append(
            f"COUNTRY_MISMATCH: billing={req.billing_country}, "
            f"shipping={req.shipping_country}"
        )
        severity = max(severity, 1)

    # Rule 6 — Address mismatch
    if not req.address_match:
        flags.append("ADDRESS_MISMATCH: billing address does not match shipping address")
        severity = max(severity, 1)

    # Rule 7 — Digital goods combo (highest risk)
    if req.is_digital_goods and req.amount_usd >= HIGH_VALUE_THRESHOLD_USD and req.account_age_days < NEW_ACCOUNT_DAYS:
        flags.append(
            "DIGITAL_GOODS_RISK: high-value digital purchase on new account "
            "(common carding pattern)"
        )
        severity = max(severity, 2)

    action = _severity_to_action(severity)
    return flags, action


# ─────────────────────────────────────────────
# Module 4 — Promotion Abuse (Coupon)
# ─────────────────────────────────────────────

def check_promo_abuse(req: PromoRequest) -> Tuple[List[str], RiskAction]:
    """
    Rules:
    1. Many accounts from same IP → flag / block
    2. Many accounts from same device → flag / block
    3. Multiple redemptions by same user → flag / block
    4. Brand-new account redeeming promo → flag
    5. Disposable email domain → flag
    """
    flags: List[str] = []
    severity = 0

    # Rule 1 — IP cluster (account farm)
    if req.accounts_same_ip > MAX_ACCOUNTS_IP_PROMO:
        flags.append(
            f"PROMO_IP_CLUSTER: {req.accounts_same_ip} accounts from IP {req.ip_address}"
        )
        severity = max(severity, 2 if req.accounts_same_ip > MAX_ACCOUNTS_IP_PROMO * 2 else 1)

    # Rule 2 — Device cluster
    if req.accounts_same_device > MAX_ACCOUNTS_DEVICE_PROMO:
        flags.append(
            f"PROMO_DEVICE_CLUSTER: {req.accounts_same_device} accounts from device "
            f"{req.device_id}"
        )
        severity = max(severity, 2 if req.accounts_same_device > MAX_ACCOUNTS_DEVICE_PROMO * 2 else 1)

    # Rule 3 — Redemption velocity per user
    if req.redemptions_by_user >= MAX_REDEMPTIONS_BLOCK:
        flags.append(
            f"PROMO_MULTI_REDEEM_BLOCK: user has redeemed {req.redemptions_by_user} times"
        )
        severity = max(severity, 2)
    elif req.redemptions_by_user >= MAX_REDEMPTIONS_WARN:
        flags.append(
            f"PROMO_MULTI_REDEEM_WARN: user has redeemed {req.redemptions_by_user} times"
        )
        severity = max(severity, 1)

    # Rule 4 — Fresh account
    if req.account_age_days == 0:
        flags.append("PROMO_FRESH_ACCOUNT: account created today — likely throwaway")
        severity = max(severity, 1)

    # Rule 5 — Disposable email
    if req.email_domain and req.email_domain.lower() in DISPOSABLE_EMAIL_DOMAINS:
        flags.append(f"PROMO_DISPOSABLE_EMAIL: domain '{req.email_domain}' is flagged")
        severity = max(severity, 1)
    elif _is_disposable_email(req.email):
        flags.append(f"PROMO_DISPOSABLE_EMAIL: {req.email.split('@')[-1]} is a disposable domain")
        severity = max(severity, 1)

    action = _severity_to_action(severity)
    return flags, action


# ─────────────────────────────────────────────
# Severity → RiskAction mapper
# ─────────────────────────────────────────────

def _severity_to_action(severity: int) -> RiskAction:
    if severity >= 2:
        return RiskAction.BLOCK
    if severity == 1:
        return RiskAction.REVIEW
    return RiskAction.ALLOW