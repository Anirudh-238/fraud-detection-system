"""
auth.py — Registration and login logic.

Handles:
- Password hashing and verification
- Auto signal assembly from HTTP request (IP, country, device)
- Fraud check on register (identity theft module)
- Fraud check on login (account takeover module)
- Database reads/writes for user and login history
"""

import uuid
from datetime import datetime, timezone, timedelta
from fastapi import Request
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from database import User, LoginLog
from geo import get_country
from schemas import LoginRequest, SignupRequest, RiskAction
from rule_engine import check_account_takeover, check_identity_theft
from ml_engine import score_request

# ─────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ─────────────────────────────────────────────
# Signal helpers
# ─────────────────────────────────────────────

def get_ip(request: Request) -> str:
    """Extract real IP from request. Checks forwarded headers first."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host or "0.0.0.0"


def get_device_id(request: Request) -> str:
    """
    Read device ID from cookie. If no cookie exists, generate one.
    The new cookie is set on the response after login/register.
    """
    return request.cookies.get("device_id") or str(uuid.uuid4())[:16]


def get_failed_attempts(db: Session, email: str) -> int:
    """Count failed login attempts for this email in the last 24 hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    return (
        db.query(LoginLog)
        .filter(
            LoginLog.email == email,
            LoginLog.action == "block",
            LoginLog.timestamp >= since,
        )
        .count()
    )


# ─────────────────────────────────────────────
# Register
# ─────────────────────────────────────────────

def register_user(
    email: str,
    full_name: str,
    password: str,
    request: Request,
    db: Session,
) -> dict:
    """
    Register a new user.
    1. Check email not already taken
    2. Assemble signup fraud signals automatically
    3. Run identity theft fraud check
    4. If not BLOCK, create the account
    5. Return verdict + message
    """

    # ── Check duplicate email ──
    existing = db.query(User).filter(User.email == email).first()
    email_seen = existing is not None

    # ── Auto-collect signals ──
    ip       = get_ip(request)
    country  = get_country(ip)
    device   = get_device_id(request)

    # Count how many accounts already exist from this IP
    accounts_from_ip = (
        db.query(User)
        .filter(User.signup_ip == ip)
        .count()
    )

    # ── Build fraud check request ──
    signup_req = SignupRequest(
        email              = email,
        full_name          = full_name,
        date_of_birth      = "1990-01-01",   # not collected — default for now
        ssn_last4          = "0000",          # not collected — default for now
        ip_address         = ip,
        email_seen_before  = email_seen,
        ssn_seen_before    = False,
        name_dob_mismatch  = False,
        accounts_from_ip   = accounts_from_ip,
    )

    # ── Run fraud check ──
    flags, rule_action = check_identity_theft(signup_req)
    fraud_score        = score_request("signup", signup_req)

    # Determine final action
    if rule_action == RiskAction.BLOCK or fraud_score >= 0.85:
        final_action = RiskAction.BLOCK
    elif fraud_score >= 0.55 or rule_action == RiskAction.REVIEW:
        final_action = RiskAction.REVIEW
    else:
        final_action = RiskAction.ALLOW

    # ── Hard block — don't create account ──
    if final_action == RiskAction.BLOCK:
        return {
            "success": False,
            "action": "block",
            "message": "Unable to complete registration. Please contact support.",
            "fraud_score": fraud_score,
            "flags": flags,
            "device_id": device,
        }

    # ── Duplicate email (but not blocked) ──
    if email_seen:
        return {
            "success": False,
            "action": "allow",
            "message": "An account with this email already exists.",
            "fraud_score": fraud_score,
            "flags": flags,
            "device_id": device,
        }

    # ── Create account ──
    user = User(
        email              = email,
        full_name          = full_name,
        hashed_password    = hash_password(password),
        signup_ip          = ip,
        signup_country     = country,
        device_id          = device,
        last_known_ip      = ip,
        last_known_country = country,
        is_flagged         = (final_action == RiskAction.REVIEW),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "success": True,
        "action": final_action.value,
        "message": "Account created successfully." if final_action == RiskAction.ALLOW
                   else "Account created. Some activity flagged for review.",
        "fraud_score": fraud_score,
        "flags": flags,
        "device_id": device,
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
        }
    }


# ─────────────────────────────────────────────
# Login
# ─────────────────────────────────────────────

def login_user(
    email: str,
    password: str,
    request: Request,
    db: Session,
) -> dict:
    """
    Authenticate a user.
    1. Look up user in DB
    2. Verify password
    3. Auto-assemble login fraud signals
    4. Run account takeover fraud check
    5. Log the attempt
    6. Return verdict + message
    """

    # ── Auto-collect signals ──
    ip        = get_ip(request)
    country   = get_country(ip)
    device    = get_device_id(request)
    ua        = request.headers.get("user-agent", "unknown")
    failed_24h = get_failed_attempts(db, email)

    # ── Look up user ──
    user = db.query(User).filter(User.email == email).first()

    # Wrong email — don't reveal user doesn't exist
    if not user or not verify_password(password, user.hashed_password):
        # Log the failed attempt
        _log_attempt(db, email, ip, country, ua, device, 0.5, "block")
        return {
            "success": False,
            "action": "block",
            "message": "Invalid email or password.",
            "fraud_score": 0.5,
            "flags": [],
            "device_id": device,
        }

    # ── Build fraud check request ──
    new_device    = (device != user.device_id) if user.device_id else False
    last_login_ts = user.last_login_at.isoformat() if user.last_login_at else None

    login_req = LoginRequest(
        user_id             = str(user.id),
        email               = email,
        ip_address          = ip,
        user_agent          = ua,
        login_timestamp     = datetime.now(timezone.utc).isoformat(),
        last_known_ip       = user.last_known_ip,
        last_login_time     = last_login_ts,
        last_known_country  = user.last_known_country,
        current_country     = country,
        failed_attempts     = failed_24h,
        new_device          = new_device,
    )

    # ── Run fraud check ──
    flags, rule_action = check_account_takeover(login_req)
    fraud_score        = score_request("login", login_req)

    # Determine final action
    if rule_action == RiskAction.BLOCK or fraud_score >= 0.85:
        final_action = RiskAction.BLOCK
    elif fraud_score >= 0.55 or rule_action == RiskAction.REVIEW:
        final_action = RiskAction.REVIEW
    else:
        final_action = RiskAction.ALLOW

    # ── Log the attempt ──
    _log_attempt(db, email, ip, country, ua, device, fraud_score, final_action.value)

    # ── Hard block ──
    if final_action == RiskAction.BLOCK:
        # Increment failed attempts on user record
        user.failed_attempts = (user.failed_attempts or 0) + 1
        db.commit()
        return {
            "success": False,
            "action": "block",
            "message": "Unable to complete login. Please contact support.",
            "fraud_score": fraud_score,
            "flags": flags,
            "device_id": device,
        }

    # ── Allow or Review — update user record ──
    user.last_login_at      = datetime.now(timezone.utc)
    user.last_known_ip      = ip
    user.last_known_country = country
    user.failed_attempts    = 0   # reset on successful login
    if final_action == RiskAction.REVIEW:
        user.is_flagged = True
    db.commit()

    return {
        "success": True,
        "action": final_action.value,
        "message": "Login successful." if final_action == RiskAction.ALLOW
                   else "Login successful. Some activity has been flagged for review.",
        "fraud_score": fraud_score,
        "flags": flags,
        "device_id": device,
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "is_flagged": user.is_flagged,
        }
    }


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _log_attempt(
    db: Session,
    email: str,
    ip: str,
    country: str | None,
    ua: str,
    device: str,
    fraud_score: float,
    action: str,
):
    """Write a login attempt to the login_logs table."""
    log = LoginLog(
        email       = email,
        ip_address  = ip,
        country     = country,
        user_agent  = ua,
        device_id   = device,
        fraud_score = fraud_score,
        action      = action,
    )
    db.add(log)
    db.commit()