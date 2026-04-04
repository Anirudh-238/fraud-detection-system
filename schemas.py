"""
schemas.py — Pydantic request/response models for all 4 fraud detection modules.
Covers: Account Takeover, Identity Theft, Payment Fraud, Promotion Abuse.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from enum import Enum


# ─────────────────────────────────────────────
# Shared enums
# ─────────────────────────────────────────────

class RiskAction(str, Enum):
    ALLOW   = "allow"
    REVIEW  = "review"
    BLOCK   = "block"


class FraudModule(str, Enum):
    ACCOUNT_TAKEOVER = "account_takeover"
    IDENTITY_THEFT   = "identity_theft"
    PAYMENT_FRAUD    = "payment_fraud"
    PROMO_ABUSE      = "promo_abuse"


# ─────────────────────────────────────────────
# Shared response wrapper
# ─────────────────────────────────────────────

class FraudResponse(BaseModel):
    """Universal response returned by every fraud check endpoint."""
    module:      FraudModule
    fraud_score: float           = Field(..., ge=0.0, le=1.0, description="ML fraud probability (0–1)")
    action:      RiskAction
    rule_flags:  List[str]       = Field(default_factory=list, description="Human-readable rule violations")
    explanation: Optional[str]   = Field(None, description="LLM explanation (Layer 3, optional)")
    request_id:  Optional[str]   = None


# ─────────────────────────────────────────────
# Module 1 — Account Takeover (Login)
# ─────────────────────────────────────────────

class LoginRequest(BaseModel):
    """Simulates a login attempt."""
    user_id:           str
    email:             EmailStr
    ip_address:        str
    user_agent:        str
    login_timestamp:   str                   = Field(..., description="ISO-8601 datetime")
    # Optional enrichment
    last_known_ip:     Optional[str]         = None
    last_login_time:   Optional[str]         = None
    last_known_country: Optional[str]        = None
    current_country:   Optional[str]         = None
    failed_attempts:   int                   = Field(0, ge=0)
    new_device:        bool                  = False

    model_config = {"json_schema_extra": {"example": {
        "user_id": "u_001",
        "email": "alice@example.com",
        "ip_address": "203.0.113.42",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0)",
        "login_timestamp": "2025-03-15T03:22:00",
        "last_known_ip": "192.168.1.1",
        "last_login_time": "2025-03-14T18:00:00",
        "last_known_country": "US",
        "current_country": "RU",
        "failed_attempts": 6,
        "new_device": True
    }}}


# ─────────────────────────────────────────────
# Module 2 — Identity Theft (Signup)
# ─────────────────────────────────────────────

class SignupRequest(BaseModel):
    """Simulates a new account registration."""
    email:           EmailStr
    full_name:       str
    date_of_birth:   str              = Field(..., description="YYYY-MM-DD")
    ssn_last4:       str              = Field(..., min_length=4, max_length=4)
    ip_address:      str
    phone_number:    Optional[str]    = None
    device_id:       Optional[str]    = None
    # Pre-computed signals (normally from your DB lookup)
    email_seen_before:   bool         = False
    ssn_seen_before:     bool         = False
    name_dob_mismatch:   bool         = False
    accounts_from_ip:    int          = Field(0, ge=0)

    model_config = {"json_schema_extra": {"example": {
        "email": "bob123@tempmail.xyz",
        "full_name": "John Smith",
        "date_of_birth": "1990-06-15",
        "ssn_last4": "0000",
        "ip_address": "10.0.0.5",
        "phone_number": "555-0100",
        "device_id": "dev_abc",
        "email_seen_before": False,
        "ssn_seen_before": True,
        "name_dob_mismatch": False,
        "accounts_from_ip": 3
    }}}


# ─────────────────────────────────────────────
# Module 3 — Payment Fraud (Checkout)
# ─────────────────────────────────────────────

class PaymentRequest(BaseModel):
    """Simulates a checkout / payment submission."""
    user_id:             str
    order_id:            str
    amount_usd:          float          = Field(..., gt=0)
    ip_address:          str
    card_bin:            str            = Field(..., min_length=6, max_length=6, description="First 6 digits of card")
    billing_country:     str
    shipping_country:    str
    email:               EmailStr
    # Velocity signals
    cards_used_24h:      int            = Field(1, ge=1)
    orders_24h:          int            = Field(1, ge=1)
    account_age_days:    int            = Field(0, ge=0)
    is_digital_goods:    bool           = False
    address_match:       bool           = True  # billing == shipping

    model_config = {"json_schema_extra": {"example": {
        "user_id": "u_042",
        "order_id": "ord_9981",
        "amount_usd": 1499.99,
        "ip_address": "185.220.101.5",
        "card_bin": "411111",
        "billing_country": "US",
        "shipping_country": "NG",
        "email": "buyer@example.com",
        "cards_used_24h": 4,
        "orders_24h": 7,
        "account_age_days": 1,
        "is_digital_goods": True,
        "address_match": False
    }}}


# ─────────────────────────────────────────────
# Module 4 — Promotion Abuse (Coupon)
# ─────────────────────────────────────────────

class PromoRequest(BaseModel):
    """Simulates a coupon / promo code redemption."""
    user_id:             str
    email:               EmailStr
    promo_code:          str
    ip_address:          str
    device_id:           Optional[str]  = None
    # Abuse signals
    accounts_same_ip:    int            = Field(1, ge=1)
    accounts_same_device: int           = Field(1, ge=1)
    redemptions_by_user: int            = Field(0, ge=0)
    account_age_days:    int            = Field(0, ge=0)
    email_domain:        Optional[str]  = None  # e.g. "tempmail.xyz"

    model_config = {"json_schema_extra": {"example": {
        "user_id": "u_777",
        "email": "freebie99@mailinator.com",
        "promo_code": "SAVE50",
        "ip_address": "10.10.10.1",
        "device_id": "dev_xyz",
        "accounts_same_ip": 8,
        "accounts_same_device": 5,
        "redemptions_by_user": 3,
        "account_age_days": 0,
        "email_domain": "mailinator.com"
    }}}


# ─────────────────────────────────────────────
# SSE alert payload (for live dashboard)
# ─────────────────────────────────────────────

class AlertEvent(BaseModel):
    """Pushed via SSE when a transaction is blocked or flagged."""
    event_type:  str           = "fraud_alert"
    module:      FraudModule
    action:      RiskAction
    fraud_score: float
    rule_flags:  List[str]
    summary:     str           # 1-line human-readable summary
    timestamp:   str           # ISO-8601