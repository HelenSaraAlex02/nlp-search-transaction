# app/data_generator.py
import os
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from app.config import DATA_PATH

def load_or_generate(filepath=DATA_PATH, num_rows=5000):
    """
    Loads dataset if exists.
    Otherwise generates and saves once.
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=["date"])
        return df

    df = generate_synthetic_transactions(num_rows)
    df.to_csv(filepath, index=False)
    return df
# ============================================================
# Configuration Constants
# ============================================================

COUNTRIES = ["US", "UK", "IN", "DE", "AE", "BH", "QA"]

COUNTRY_CURRENCY_MAP = {
    "US": "USD",
    "UK": "GBP",
    "IN": "INR",
    "DE": "EUR",
    "AE": "AED",
    "BH": "BHD",
    "QA": "QAR"
}

COUNTRY_NAME_MAP = {
    "US": "United States",
    "UK": "United Kingdom",
    "IN": "India",
    "DE": "Germany",
    "AE": "United Arab Emirates",
    "BH": "Bahrain",
    "QA": "Qatar"
}

STATUSES = ["COMPLETED", "FAILED", "PENDING", "RETURNED"]
FAILURE_REASONS = ["Timeout", "Insufficient Funds", "Compliance Hold"]
TRANSACTION_TYPES = ["SALARY", "TRADE", "TRANSFER", "FEE"]
CUSTOMER_SEGMENTS = ["RETAIL", "SME", "CORPORATE", "FI"]

fake = Faker()

# ============================================================
# Helper Functions
# ============================================================

def random_date_last_year():
    today = datetime.today()
    past_date = today - timedelta(days=365)
    return past_date + timedelta(days=random.randint(0, 365))


def determine_payment_rail(sender_country, receiver_country, amount):
    if sender_country != receiver_country:
        return "SWIFT"
    if amount < 10000:
        return "RTP"
    return "RTGS"


# ============================================================
# Core Data Generation Function
# ============================================================

def generate_synthetic_transactions(num_rows: int = 5000, seed: int = 42):
    """
    Generates synthetic transaction dataset with feature engineering
    and text descriptions for embedding.
    """

    np.random.seed(seed)
    random.seed(seed)

    data = []

    for i in range(num_rows):

        transaction_id = f"TXN{i:06d}"
        date = random_date_last_year()

        sender_country = random.choice(COUNTRIES)
        receiver_country = random.choice(COUNTRIES)

        currency = COUNTRY_CURRENCY_MAP[receiver_country]
        amount = round(np.random.uniform(50, 50000), 2)

        payment_rail = determine_payment_rail(
            sender_country,
            receiver_country,
            amount
        )

        status = random.choice(STATUSES)

        failure_reason = None
        if status == "FAILED":
            failure_reason = random.choice(FAILURE_REASONS)

        data.append({
            "transaction_id": transaction_id,
            "date": date,
            "amount": amount,
            "currency": currency,
            "sender_name": fake.company(),
            "receiver_name": fake.company(),
            "receiver_country": receiver_country,
            "payment_rail": payment_rail,
            "status": status,
            "failure_reason": failure_reason,
            "transaction_type": random.choice(TRANSACTION_TYPES),
            "customer_segment": random.choice(CUSTOMER_SEGMENTS),
        })

    df = pd.DataFrame(data)

    # ============================================================
    # Feature Engineering
    # ============================================================

    df["receiver_country_name"] = df["receiver_country"].map(COUNTRY_NAME_MAP)

    df["transaction_scope"] = df["payment_rail"].apply(
        lambda x: "cross-border" if x == "SWIFT" else "domestic"
    )

    df["value_category"] = df["amount"].apply(
        lambda x: "high value" if x >= 10000 else "low value"
    )

    # ============================================================
    # Build Embedding Text
    # ============================================================

    df["text_description"] = df.apply(build_transaction_sentence, axis=1)

    return df


# ============================================================
# Sentence Builder (For Embeddings)
# ============================================================

def build_transaction_sentence(row):

    sentence = (
        f"A {row['value_category']} {row['transaction_scope']} "
        f"{row['payment_rail']} transfer of {row['amount']} {row['currency']} "
        f"was sent to {row['receiver_name']} in "
        f"{row['receiver_country_name']} on "
        f"{row['date'].strftime('%Y-%m-%d')}. "
        f"The transaction type was {row['transaction_type']} "
        f"for a {row['customer_segment']} customer "
        f"with status as {row['status']}."
    )

    if row["status"] == "FAILED" and pd.notnull(row["failure_reason"]):
        sentence += f" Failure reason: {row['failure_reason']}."

    return sentence


# ============================================================
# Optional CSV Export
# ============================================================

def save_to_csv(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)

