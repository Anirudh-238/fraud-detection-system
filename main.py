"""
main.py — FastAPI application: routes, middleware, SSE live dashboard.

Start server:
    uvicorn main:app --reload

Endpoints:
    POST /check/login
    POST /check/signup
    POST /check/payment
    POST /check/promo
    GET  /dashboard/stream   <- SSE live alerts
    GET  /health
    GET  /docs               <- Auto-generated Swagger UI
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from schemas import (
    LoginRequest, SignupRequest, PaymentRequest, PromoRequest,
    FraudResponse, AlertEvent, RiskAction, FraudModule
)
from rule_engine import (
    check_account_takeover, check_identity_theft,
    check_payment_fraud, check_promo_abuse
)
from ml_engine import load_all_models, score_request

from database import init_db, get_db
from auth import register_user, login_user
from sqlalchemy.orm import Session
from fastapi import Depends

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="4-module fraud detection system with rule engine + ML scoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# SSE alert queue (in-memory broadcast)
# ─────────────────────────────────────────────

alert_subscribers: list[asyncio.Queue] = []


async def broadcast_alert(alert: AlertEvent):
    """Push an alert to all connected SSE dashboard clients."""
    data = alert.model_dump_json()
    for queue in alert_subscribers:
        await queue.put(data)


async def sse_event_generator(request: Request) -> AsyncGenerator[str, None]:
    """Generator that streams alerts to a connected dashboard client."""
    queue: asyncio.Queue = asyncio.Queue()
    alert_subscribers.append(queue)
    try:
        # Send a heartbeat immediately on connect
        yield "data: {\"event_type\": \"connected\"}\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                data = await asyncio.wait_for(queue.get(), timeout=15.0)
                yield f"data: {data}\n\n"
            except asyncio.TimeoutError:
                # Send keep-alive ping every 15s
                yield "data: {\"event_type\": \"ping\"}\n\n"
    finally:
        alert_subscribers.remove(queue)


# ─────────────────────────────────────────────
# Startup — load ML models
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("\n[Startup] Loading ML models...")
    load_all_models()
    init_db()
    print("[Startup] All models loaded. Server ready.\n")


# ─────────────────────────────────────────────
# Helper — combine rule + ML into final response
# ─────────────────────────────────────────────

def _build_response(
    module: FraudModule,
    rule_flags: list,
    rule_action: RiskAction,
    fraud_score: float,
    request_id: str,
) -> FraudResponse:
    """
    Combine rule engine decision + ML score into a final action.
    Rule engine can hard-block regardless of ML score.
    ML score can escalate a REVIEW to BLOCK if score is very high.
    """
    # Rule engine hard block always wins
    if rule_action == RiskAction.BLOCK:
        final_action = RiskAction.BLOCK
    # ML score >= 0.85 escalates to block even if rules said review/allow
    elif fraud_score >= 0.85:
        final_action = RiskAction.BLOCK
        rule_flags.append(f"ML_HIGH_SCORE: fraud probability {fraud_score:.2f} exceeds threshold 0.85")
    # ML score >= 0.55 at minimum triggers review
    elif fraud_score >= 0.55 or rule_action == RiskAction.REVIEW:
        final_action = RiskAction.REVIEW
    else:
        final_action = RiskAction.ALLOW

    return FraudResponse(
        module=module,
        fraud_score=fraud_score,
        action=final_action,
        rule_flags=rule_flags,
        request_id=request_id,
    )


async def _maybe_alert(response: FraudResponse, summary: str):
    """If the result is BLOCK or REVIEW, broadcast an SSE alert."""
    if response.action in (RiskAction.BLOCK, RiskAction.REVIEW):
        alert = AlertEvent(
            module=response.module,
            action=response.action,
            fraud_score=response.fraud_score,
            rule_flags=response.rule_flags,
            summary=summary,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        await broadcast_alert(alert)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/dashboard/stream", tags=["Dashboard"])
async def dashboard_stream(request: Request):
    """SSE endpoint — connect here to receive live fraud alerts."""
    return StreamingResponse(
        sse_event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/check/login", response_model=FraudResponse, tags=["Fraud Checks"])
async def check_login(req: LoginRequest):
    """Check a login attempt for account takeover signals."""
    request_id = str(uuid.uuid4())[:8]

    # Layer 1 — rules
    flags, rule_action = check_account_takeover(req)

    # Layer 2 — ML score
    fraud_score = score_request("login", req)

    # Combine
    response = _build_response(
        FraudModule.ACCOUNT_TAKEOVER, flags, rule_action, fraud_score, request_id
    )

    # Alert dashboard if needed
    await _maybe_alert(
        response,
        f"Login attempt from {req.ip_address} — score {fraud_score:.2f} — {response.action}"
    )

    return response


@app.post("/check/signup", response_model=FraudResponse, tags=["Fraud Checks"])
async def check_signup(req: SignupRequest):
    """Check a signup attempt for identity theft signals."""
    request_id = str(uuid.uuid4())[:8]

    flags, rule_action = check_identity_theft(req)
    fraud_score = score_request("signup", req)

    response = _build_response(
        FraudModule.IDENTITY_THEFT, flags, rule_action, fraud_score, request_id
    )

    await _maybe_alert(
        response,
        f"Signup from {req.email} — score {fraud_score:.2f} — {response.action}"
    )

    return response


@app.post("/check/payment", response_model=FraudResponse, tags=["Fraud Checks"])
async def check_payment(req: PaymentRequest):
    """Check a payment for fraud signals."""
    request_id = str(uuid.uuid4())[:8]

    flags, rule_action = check_payment_fraud(req)
    fraud_score = score_request("payment", req)

    response = _build_response(
        FraudModule.PAYMENT_FRAUD, flags, rule_action, fraud_score, request_id
    )

    await _maybe_alert(
        response,
        f"Payment ${req.amount_usd:.2f} from {req.ip_address} — score {fraud_score:.2f} — {response.action}"
    )

    return response


@app.post("/check/promo", response_model=FraudResponse, tags=["Fraud Checks"])
async def check_promo(req: PromoRequest):
    """Check a promo redemption for abuse signals."""
    request_id = str(uuid.uuid4())[:8]

    flags, rule_action = check_promo_abuse(req)
    fraud_score = score_request("promo", req)

    response = _build_response(
        FraudModule.PROMO_ABUSE, flags, rule_action, fraud_score, request_id
    )

    await _maybe_alert(
        response,
        f"Promo '{req.promo_code}' by {req.email} — score {fraud_score:.2f} — {response.action}"
    )

    return response



# ─────────────────────────────────────────────
# Auth routes (Day 4)
# ─────────────────────────────────────────────

@app.post("/auth/register", tags=["Auth"])
async def register(req: Request, db: Session = Depends(get_db)):
    body = await req.json()
    result = register_user(
        email     = body.get("email"),
        full_name = body.get("full_name"),
        password  = body.get("password"),
        request   = req,
        db        = db,
    )
    if result["action"] == "block":
        response = JSONResponse(content=result, status_code=403)
    else:
        response = JSONResponse(content=result, status_code=200)
    response.set_cookie("device_id", result["device_id"], max_age=60*60*24*365)
    return response


@app.post("/auth/login", tags=["Auth"])
async def login(req: Request, db: Session = Depends(get_db)):
    body = await req.json()
    result = login_user(
        email    = body.get("email"),
        password = body.get("password"),
        request  = req,
        db       = db,
    )
    if not result["success"]:
        response = JSONResponse(content=result, status_code=401)
    else:
        response = JSONResponse(content=result, status_code=200)
    response.set_cookie("device_id", result["device_id"], max_age=60*60*24*365)
    return response

# ─────────────────────────────────────────────
# Serve frontend (Day 3)
# ─────────────────────────────────────────────

try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
except RuntimeError:
    # frontend/ folder doesn't exist yet (Day 3)
    pass