"""
train.py — Layer 2: Train one XGBoost model per fraud module.

Reads CSVs from data/, trains, evaluates, and saves models to models/.

Usage:
    python train.py                  # trains all 4 models
    python train.py --module login   # trains only login model
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

DATA_DIR   = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Feature columns per module
# These must match exactly what data_gen.py produces
# ─────────────────────────────────────────────

FEATURES = {
    "login": [
        "failed_attempts", "new_device", "ip_changed",
        "country_changed", "current_high_risk", "hour_of_day", "odd_hour"
    ],
    "signup": [
        "ssn_seen_before", "email_seen_before", "name_dob_mismatch",
        "accounts_from_ip", "disposable_email"
    ],
    "payment": [
        "amount_usd", "cards_used_24h", "orders_24h",
        "account_age_days", "country_mismatch", "address_match", "is_digital_goods"
    ],
    "promo": [
        "accounts_same_ip", "accounts_same_device",
        "redemptions_by_user", "account_age_days", "disposable_email"
    ],
}

CSV_FILES = {
    "login":   "login_data.csv",
    "signup":  "signup_data.csv",
    "payment": "payment_data.csv",
    "promo":   "promo_data.csv",
}


# ─────────────────────────────────────────────
# Train one module
# ─────────────────────────────────────────────

def train_module(module: str):
    print(f"\n{'='*50}")
    print(f"  Training: {module.upper()}")
    print(f"{'='*50}")

    # Load data
    csv_path = os.path.join(DATA_DIR, CSV_FILES[module])
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found. Run data_gen.py first.")
        return

    df = pd.read_csv(csv_path)
    features = FEATURES[module]
    X = df[features]
    y = df["is_fraud"]

    print(f"  Rows: {len(df)}  |  Fraud: {y.sum()} ({y.mean()*100:.1f}%)")

    # Train / test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # handle class imbalance
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n  AUC-ROC Score: {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # Feature importance
    importance = dict(zip(features, model.feature_importances_))
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("  Top Features:")
    for feat, score in top:
        bar = "█" * int(score * 40)
        print(f"    {feat:<25} {bar} {score:.4f}")

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{module}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":    model,
            "features": features,
            "module":   module,
        }, f)

    print(f"\n  Saved → {model_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost fraud detection models")
    parser.add_argument("--module", choices=list(FEATURES.keys()), default=None,
                        help="Which module to train (default: all)")
    args = parser.parse_args()

    modules = [args.module] if args.module else list(FEATURES.keys())

    for module in modules:
        train_module(module)

    print(f"\n✅ All models saved to ./{MODELS_DIR}/")


if __name__ == "__main__":
    main()