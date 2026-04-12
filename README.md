# FraudShield — Real-Time Fraud Detection System

A full-stack fraud detection API built as a B.Tech final year project. Combines a deterministic rule engine with XGBoost ML models across four fraud modules, with a live SSE dashboard for real-time monitoring.

---

## Architecture

```
Request
   │
   ├── Layer 1: Rule Engine (deterministic, microseconds)
   │     └── Hard-coded thresholds → BLOCK / REVIEW / ALLOW
   │
   ├── Layer 2: XGBoost ML (probabilistic, per-module model)
   │     └── Fraud probability score 0.0 – 1.0
   │
   └── Layer 3: Human-in-the-Loop
         └── Live SSE dashboard → analyst review
```

**Decision logic:**
- Rule engine BLOCK → always BLOCK (ML cannot override)
- ML score ≥ 0.85 → escalate to BLOCK
- ML score ≥ 0.55 OR rule REVIEW → REVIEW
- Otherwise → ALLOW

---

## Fraud Modules

| Module | Endpoint | Detects |
|--------|----------|---------|
| Account Takeover | `POST /check/login` | Brute-force, impossible travel, new device, odd-hour logins |
| Identity Theft | `POST /check/signup` | Synthetic IDs, SSN reuse, disposable emails, IP account farms |
| Payment Fraud | `POST /check/payment` | Carding, card velocity, country mismatch, digital goods risk |
| Promo Abuse | `POST /check/promo` | IP/device clusters, multi-redemption, throwaway accounts |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| ML | XGBoost, scikit-learn, NumPy, Pandas |
| Database | SQLite via SQLAlchemy ORM |
| Auth | bcrypt password hashing via passlib |
| Geo | ip-api.com (free, no key required) |
| Frontend | Vanilla HTML/CSS/JS |
| Realtime | Server-Sent Events (SSE) |

---

## Project Structure

```
fraudshield/
├── main.py           # FastAPI app — all routes + SSE dashboard
├── rule_engine.py    # Layer 1 — deterministic fraud rules (4 modules)
├── ml_engine.py      # Layer 2 — XGBoost model loader + scorer
├── auth.py           # Real auth — register/login with fraud checks baked in
├── schemas.py        # Pydantic request/response models
├── database.py       # SQLAlchemy models + SQLite setup
├── geo.py            # IP → country lookup (ip-api.com)
├── data_gen.py       # Synthetic fraud dataset generator
├── train.py          # XGBoost model trainer (one model per module)
├── requirements.txt
├── .gitignore
└── frontend/
    ├── index.html    # Landing page
    ├── dashboard.html# Live SSE alert dashboard + charts
    ├── api-docs.html # Developer API reference
    ├── login.html    # Module 01 demo — Account Takeover
    ├── signup.html   # Module 02 demo — Identity Theft
    ├── payment.html  # Module 03 demo — Payment Fraud
    ├── promo.html    # Module 04 demo — Promo Abuse
    └── style.css     # Shared design system
```

---

## Setup & Run

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd fraudshield
pip install -r requirements.txt
```

### 2. Generate training data

```bash
python data_gen.py
# Creates: data/login_data.csv, signup_data.csv, payment_data.csv, promo_data.csv
```

### 3. Train the ML models

```bash
python train.py
# Creates: models/login_model.pkl, signup_model.pkl, payment_model.pkl, promo_model.pkl
# Prints AUC-ROC score and top features for each model
```

### 4. Start the server

```bash
uvicorn main:app --reload
```

Server runs at `http://localhost:8000`

### 5. Open the frontend

Open `frontend/index.html` in your browser — or navigate to `http://localhost:8000` if you've placed the frontend folder in the project root (FastAPI serves it automatically).

---

## API Reference

### Base URL
```
http://localhost:8000
```

### Auth Endpoints

```
POST /auth/register   — Create a real account (fraud-checked)
POST /auth/login      — Authenticate (fraud-checked, logged)
```

### Fraud Check Endpoints

```
POST /check/login     — Account takeover check
POST /check/signup    — Identity theft check
POST /check/payment   — Payment fraud check
POST /check/promo     — Promo abuse check
```

### System

```
GET  /dashboard/stream — SSE live alert stream
GET  /health           — Server status
GET  /docs             — Swagger UI (auto-generated)
```

### Example Request

```python
import requests

response = requests.post("http://localhost:8000/check/payment", json={
    "user_id":          "u_042",
    "order_id":         "ord_9981",
    "amount_usd":       1499.99,
    "ip_address":       "185.220.101.5",
    "card_bin":         "411111",
    "billing_country":  "US",
    "shipping_country": "NG",
    "email":            "buyer@example.com",
    "cards_used_24h":   5,
    "account_age_days": 1,
    "is_digital_goods": False,
    "address_match":    False,
})

print(response.json())
# {
#   "action": "block",
#   "fraud_score": 0.94,
#   "rule_flags": [
#     "COUNTRY_MISMATCH: billing=US, shipping=NG",
#     "CARD_VELOCITY: 5 different cards used in 24h",
#     "HIGH_VALUE_NEW_ACCOUNT: $1499.99 on account 1 day old"
#   ],
#   "module": "payment_fraud",
#   "request_id": "a3f9b2c1"
# }
```

### SSE Dashboard Client

```javascript
const es = new EventSource("http://localhost:8000/dashboard/stream");

es.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.event_type === "ping") return;
  console.log(data.action, data.fraud_score, data.rule_flags);
};
```

---

## Demo Scenarios

Each frontend page has preset buttons that populate realistic fraud scenarios:

**Account Takeover** (`/login`)
- ✓ Normal login — known device, same country, 0 failed attempts
- ⚠ Brute force — 9 failed attempts, new device, country changed US→RU
- ⚠ Impossible travel — country changed US→CN, new device

**Identity Theft** (`/signup`)
- ✓ Legit signup — real email, 1 account from IP
- ⚠ Disposable email — mailinator.com domain
- ⚠ Account farm — 8 accounts from same IP
- ⚠ Synthetic identity — SSN reuse + name/DOB mismatch

**Payment Fraud** (`/payment`)
- ✓ Normal purchase — $49.99, matched countries, old account
- ⚠ Carding — $1,499, US billing → NG shipping, 5 cards/24h, 1-day account
- ⚠ Order velocity — 12 orders in 24h
- ⚠ Digital goods risk — $2,499 digital goods, 0-day account

**Promo Abuse** (`/promo`)
- ✓ Legit redemption — 1 account, 0 prior redemptions
- ⚠ IP cluster — 9 accounts from same IP
- ⚠ Multi-redeem — user redeemed 4 times
- ⚠ Full abuse combo — disposable email + IP cluster + multi-redeem

---

## Inspired By

- [Sift](https://sift.com) — machine learning fraud detection
- [Stripe Radar](https://stripe.com/radar) — rule + ML hybrid architecture
- [Kount](https://kount.com) — identity trust signals

---

## Authors

**Anirudh** — B.Tech Final Year Project