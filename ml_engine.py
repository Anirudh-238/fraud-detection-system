"""
ml_engine.py — Layer 2: Load trained XGBoost models and score requests.

Returns a fraud probability (0.0 = definitely legit, 1.0 = definitely fraud).
Called by main.py after the rule engine has already run.
"""

import os
import pickle
import numpy as np
from typing import Dict

MODELS_DIR = "models"

# ─────────────────────────────────────────────
# Load all models once at startup
# ─────────────────────────────────────────────

_models: Dict[str, dict] = {}

def load_all_models():
    """Load all 4 trained models into memory. Call this once on app startup."""
    modules = ["login", "signup", "payment", "promo"]
    for module in modules:
        path = os.path.join(MODELS_DIR, f"{module}_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                _models[module] = pickle.load(f)
            print(f"  [ML] Loaded model: {module}")
        else:
            print(f"  [ML] WARNING: No model found for '{module}' at {path}")
            print(f"       Run train.py first.")


def _get_model(module: str):
    if module not in _models:
        raise RuntimeError(
            f"Model '{module}' not loaded. "
            f"Make sure load_all_models() was called at startup "
            f"and train.py has been run."
        )
    return _models[module]


# ─────────────────────────────────────────────
# Feature extractors
# Convert Pydantic request objects → flat feature dict
# Must match the columns in data_gen.py exactly
# ─────────────────────────────────────────────

def _extract_login_features(req) -> dict:
    from datetime import datetime
    hour = -1
    try:
        hour = datetime.fromisoformat(req.login_timestamp).hour
    except Exception:
        pass

    last_country    = (req.last_known_country or "").upper()
    current_country = (req.current_country or "").upper()
    high_risk       = {"NG", "RU", "CN"}

    return {
        "failed_attempts":   req.failed_attempts,
        "new_device":        int(req.new_device),
        "ip_changed":        int(bool(req.last_known_ip) and req.ip_address != req.last_known_ip),
        "country_changed":   int(bool(last_country) and last_country != current_country),
        "current_high_risk": int(current_country in high_risk),
        "hour_of_day":       hour if hour >= 0 else 12,
        "odd_hour":          int(hour in range(1, 5)),
    }


def _extract_signup_features(req) -> dict:
    disposable = {
        "mailinator.com", "tempmail.xyz", "throwaway.email",
        "guerrillamail.com", "yopmail.com", "sharklasers.com",
        "trashmail.com", "maildrop.cc",
    }
    domain = req.email.split("@")[-1].lower()
    return {
        "ssn_seen_before":   int(req.ssn_seen_before),
        "email_seen_before": int(req.email_seen_before),
        "name_dob_mismatch": int(req.name_dob_mismatch),
        "accounts_from_ip":  req.accounts_from_ip,
        "disposable_email":  int(domain in disposable),
    }


def _extract_payment_features(req) -> dict:
    return {
        "amount_usd":       req.amount_usd,
        "cards_used_24h":   req.cards_used_24h,
        "orders_24h":       req.orders_24h,
        "account_age_days": req.account_age_days,
        "country_mismatch": int(req.billing_country.upper() != req.shipping_country.upper()),
        "address_match":    int(req.address_match),
        "is_digital_goods": int(req.is_digital_goods),
    }


def _extract_promo_features(req) -> dict:
    disposable = {
        "mailinator.com", "tempmail.xyz", "throwaway.email",
        "guerrillamail.com", "yopmail.com", "sharklasers.com",
        "trashmail.com", "maildrop.cc",
    }
    domain = req.email.split("@")[-1].lower()
    return {
        "accounts_same_ip":     req.accounts_same_ip,
        "accounts_same_device": req.accounts_same_device,
        "redemptions_by_user":  req.redemptions_by_user,
        "account_age_days":     req.account_age_days,
        "disposable_email":     int(domain in disposable),
    }


FEATURE_EXTRACTORS = {
    "login":   _extract_login_features,
    "signup":  _extract_signup_features,
    "payment": _extract_payment_features,
    "promo":   _extract_promo_features,
}


# ─────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────

def score_request(module: str, request) -> float:
    """
    Score a request using the trained XGBoost model.

    Args:
        module:  One of "login", "signup", "payment", "promo"
        request: The Pydantic request object from schemas.py

    Returns:
        float: Fraud probability between 0.0 and 1.0
    """
    model_data = _get_model(module)
    model      = model_data["model"]
    features   = model_data["features"]

    # Extract features from request
    extractor    = FEATURE_EXTRACTORS[module]
    feature_dict = extractor(request)

    # Build feature vector in correct column order
    X = np.array([[feature_dict[f] for f in features]])

    # Get fraud probability
    fraud_score = float(model.predict_proba(X)[0][1])
    return round(fraud_score, 4)


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading models...")
    load_all_models()
    print(f"\nLoaded modules: {list(_models.keys())}")
    print("\nml_engine.py is ready.")