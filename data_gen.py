"""
data_gen.py — Synthetic fraud dataset generator.
Produces labeled CSV files for training the XGBoost model (Day 2).

Usage:
    python data_gen.py                    # generates all 4 datasets
    python data_gen.py --module login     # only account takeover data
    python data_gen.py --samples 5000     # custom sample count (default 2000)

Output files (in ./data/):
    login_data.csv
    signup_data.csv
    payment_data.csv
    promo_data.csv

Label column: `is_fraud` (1 = fraud, 0 = legit)
Fraud ratio: ~30% fraud, 70% legit (class imbalance intentional — realistic).
"""

import os
import argparse
import random
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DISPOSABLE_DOMAINS = [
    "mailinator.com", "tempmail.xyz", "throwaway.email",
    "guerrillamail.com", "yopmail.com", "sharklasers.com",
    "trashmail.com", "maildrop.cc",
]
LEGIT_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com",
    "hotmail.com", "icloud.com", "proton.me",
]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "IN", "AU", "NG", "RU", "CN", "BR"]
HIGH_RISK_COUNTRIES = ["NG", "RU", "CN"]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _random_ip(private: bool = False) -> str:
    if private:
        return f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
    return fake.ipv4_public()


def _random_hour(fraud: bool = False) -> int:
    """Fraud logins skew toward 1–4 AM."""
    if fraud:
        return random.choices(
            range(24),
            weights=[1,5,5,5,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        )[0]
    return random.choices(range(24), weights=[1,1,1,1,1,2,3,4,5,5,5,5,5,5,4,4,4,3,3,2,2,2,1,1])[0]


def _random_timestamp(fraud: bool = False) -> str:
    base = datetime.now() - timedelta(days=random.randint(0, 30))
    hour = _random_hour(fraud)
    return base.replace(hour=hour, minute=random.randint(0, 59)).isoformat()


# ─────────────────────────────────────────────
# Module 1 — Login / Account Takeover data
# ─────────────────────────────────────────────

def gen_login_row(fraud: bool) -> dict:
    last_country = random.choice(COUNTRIES)

    if fraud:
        # Inject fraud signals
        current_country = random.choice(HIGH_RISK_COUNTRIES + [c for c in COUNTRIES if c != last_country])
        failed_attempts = random.randint(4, 15)
        new_device      = random.choice([True, True, False])
        ip_changed      = True
    else:
        current_country = last_country if random.random() > 0.1 else random.choice(COUNTRIES)
        failed_attempts = random.choices([0, 1, 2, 3], weights=[70, 15, 10, 5])[0]
        new_device      = random.random() < 0.1
        ip_changed      = random.random() < 0.15

    hour = _random_hour(fraud)
    return {
        "failed_attempts":      failed_attempts,
        "new_device":           int(new_device),
        "ip_changed":           int(ip_changed),
        "country_changed":      int(current_country != last_country),
        "current_high_risk":    int(current_country in HIGH_RISK_COUNTRIES),
        "hour_of_day":          hour,
        "odd_hour":             int(hour in range(1, 5)),
        "is_fraud":             int(fraud),
    }


def gen_login_dataset(n: int = 2000) -> pd.DataFrame:
    n_fraud = int(n * 0.30)
    rows = [gen_login_row(True) for _ in range(n_fraud)] + \
           [gen_login_row(False) for _ in range(n - n_fraud)]
    df = pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Module 2 — Signup / Identity Theft data
# ─────────────────────────────────────────────

def gen_signup_row(fraud: bool) -> dict:
    if fraud:
        email_domain      = random.choice(DISPOSABLE_DOMAINS + LEGIT_DOMAINS)
        ssn_seen_before   = random.random() < 0.6
        email_seen_before = random.random() < 0.5
        name_dob_mismatch = random.random() < 0.4
        accounts_from_ip  = random.randint(3, 12)
        disposable_email  = email_domain in DISPOSABLE_DOMAINS
    else:
        email_domain      = random.choice(LEGIT_DOMAINS)
        ssn_seen_before   = random.random() < 0.02
        email_seen_before = random.random() < 0.03
        name_dob_mismatch = random.random() < 0.05
        accounts_from_ip  = random.choices([1, 2, 3], weights=[80, 15, 5])[0]
        disposable_email  = False

    return {
        "ssn_seen_before":   int(ssn_seen_before),
        "email_seen_before": int(email_seen_before),
        "name_dob_mismatch": int(name_dob_mismatch),
        "accounts_from_ip":  accounts_from_ip,
        "disposable_email":  int(disposable_email),
        "is_fraud":          int(fraud),
    }


def gen_signup_dataset(n: int = 2000) -> pd.DataFrame:
    n_fraud = int(n * 0.30)
    rows = [gen_signup_row(True) for _ in range(n_fraud)] + \
           [gen_signup_row(False) for _ in range(n - n_fraud)]
    return pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)


# ─────────────────────────────────────────────
# Module 3 — Payment Fraud data
# ─────────────────────────────────────────────

def gen_payment_row(fraud: bool) -> dict:
    if fraud:
        amount           = round(random.uniform(400, 5000), 2)
        cards_used_24h   = random.randint(3, 8)
        orders_24h       = random.randint(5, 20)
        account_age_days = random.randint(0, 7)
        country_mismatch = int(random.random() < 0.7)
        address_match    = int(random.random() < 0.2)
        is_digital       = int(random.random() < 0.6)
    else:
        amount           = round(random.choices(
            [random.uniform(5, 100), random.uniform(100, 500), random.uniform(500, 1500)],
            weights=[60, 30, 10]
        )[0], 2)
        cards_used_24h   = random.choices([1, 2, 3], weights=[80, 15, 5])[0]
        orders_24h       = random.choices([1, 2, 3, 4], weights=[60, 25, 10, 5])[0]
        account_age_days = random.randint(7, 1000)
        country_mismatch = int(random.random() < 0.1)
        address_match    = int(random.random() > 0.05)
        is_digital       = int(random.random() < 0.3)

    return {
        "amount_usd":        amount,
        "cards_used_24h":    cards_used_24h,
        "orders_24h":        orders_24h,
        "account_age_days":  account_age_days,
        "country_mismatch":  country_mismatch,
        "address_match":     address_match,
        "is_digital_goods":  is_digital,
        "is_fraud":          int(fraud),
    }


def gen_payment_dataset(n: int = 2000) -> pd.DataFrame:
    n_fraud = int(n * 0.30)
    rows = [gen_payment_row(True) for _ in range(n_fraud)] + \
           [gen_payment_row(False) for _ in range(n - n_fraud)]
    return pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)


# ─────────────────────────────────────────────
# Module 4 — Promo Abuse data
# ─────────────────────────────────────────────

def gen_promo_row(fraud: bool) -> dict:
    if fraud:
        accounts_same_ip     = random.randint(4, 15)
        accounts_same_device = random.randint(3, 10)
        redemptions          = random.randint(2, 8)
        account_age_days     = random.randint(0, 3)
        disposable_email     = int(random.random() < 0.7)
    else:
        accounts_same_ip     = random.choices([1, 2, 3], weights=[75, 20, 5])[0]
        accounts_same_device = random.choices([1, 2, 3], weights=[80, 15, 5])[0]
        redemptions          = random.choices([0, 1], weights=[85, 15])[0]
        account_age_days     = random.randint(10, 800)
        disposable_email     = int(random.random() < 0.05)

    return {
        "accounts_same_ip":     accounts_same_ip,
        "accounts_same_device": accounts_same_device,
        "redemptions_by_user":  redemptions,
        "account_age_days":     account_age_days,
        "disposable_email":     disposable_email,
        "is_fraud":             int(fraud),
    }


def gen_promo_dataset(n: int = 2000) -> pd.DataFrame:
    n_fraud = int(n * 0.30)
    rows = [gen_promo_row(True) for _ in range(n_fraud)] + \
           [gen_promo_row(False) for _ in range(n - n_fraud)]
    return pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

GENERATORS = {
    "login":   (gen_login_dataset,   "login_data.csv"),
    "signup":  (gen_signup_dataset,  "signup_data.csv"),
    "payment": (gen_payment_dataset, "payment_data.csv"),
    "promo":   (gen_promo_dataset,   "promo_data.csv"),
}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud datasets")
    parser.add_argument("--module",  choices=list(GENERATORS.keys()), default=None,
                        help="Which module to generate (default: all)")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of rows per dataset (default: 2000)")
    args = parser.parse_args()

    modules = [args.module] if args.module else list(GENERATORS.keys())

    for mod in modules:
        gen_fn, filename = GENERATORS[mod]
        print(f"  Generating {args.samples} rows for '{mod}'...", end=" ")
        df = gen_fn(args.samples)
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        fraud_pct = df["is_fraud"].mean() * 100
        print(f"saved → {path}  (fraud={fraud_pct:.1f}%)")

    print("\n✅ All datasets ready in ./data/")


if __name__ == "__main__":
    main()