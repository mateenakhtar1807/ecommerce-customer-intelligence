"""
generate_dataset.py
--------------------
Generates a synthetic but behaviourally-realistic e-commerce customer/session
dataset for the E-Commerce Customer Intelligence & Purchase Prediction System.

IMPORTANT: This dataset is SYNTHETIC. It is generated programmatically to
*simulate* realistic e-commerce behaviour (correlations between engagement
signals and purchase outcome, noise, class imbalance) so that the ML models
trained on it behave the way models trained on real e-commerce data would.
It is not scraped or copied from any real company.

Run:
    python generate_dataset.py

Output:
    data/ecommerce_customer_data.csv
    data/products.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

RANDOM_SEED = 42
N_RECORDS = 12000

np.random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune",
          "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"]
DEVICE_TYPES = ["Mobile", "Desktop", "Tablet"]
DEVICE_WEIGHTS = [0.62, 0.30, 0.08]
TRAFFIC_SOURCES = ["Organic Search", "Paid Ads", "Social Media", "Direct", "Email", "Referral"]
TRAFFIC_WEIGHTS = [0.28, 0.22, 0.20, 0.15, 0.09, 0.06]
# Baseline conversion multiplier per traffic source (Email/Direct convert better - warm traffic)
TRAFFIC_CONV_BOOST = {
    "Organic Search": 0.0, "Paid Ads": -0.05, "Social Media": -0.08,
    "Direct": 0.08, "Email": 0.12, "Referral": 0.04
}
PRODUCT_CATEGORIES = ["Rings", "Bracelets", "Necklaces", "Earrings", "Pendants",
                       "Anklets", "Nose Pins", "Couple Sets"]
GENDERS = ["Female", "Male", "Other"]
GENDER_WEIGHTS = [0.58, 0.39, 0.03]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_dataset(n=N_RECORDS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    customer_id = np.array([f"CUST{100000+i}" for i in range(n)])
    session_id = np.array([f"SESS{200000+i}" for i in range(n)])

    # Timestamps over the last 180 days
    start_date = datetime(2026, 2, 1)
    random_days = rng.integers(0, 180, size=n)
    random_seconds = rng.integers(0, 86400, size=n)
    timestamp = [start_date + timedelta(days=int(d), seconds=int(s))
                 for d, s in zip(random_days, random_seconds)]

    age = np.clip(rng.normal(28, 7, n), 16, 65).astype(int)
    gender = rng.choice(GENDERS, size=n, p=GENDER_WEIGHTS)
    city = rng.choice(CITIES, size=n)
    device_type = rng.choice(DEVICE_TYPES, size=n, p=DEVICE_WEIGHTS)
    traffic_source = rng.choice(TRAFFIC_SOURCES, size=n, p=TRAFFIC_WEIGHTS)

    # Engagement signals - session_duration in minutes (log-normal, realistic long tail)
    session_duration = np.round(np.clip(rng.lognormal(mean=1.4, sigma=0.7, size=n), 0.2, 60), 2)

    pages_viewed = np.clip((rng.poisson(4, n) + session_duration / 3).astype(int), 1, 60)
    product_views = np.clip((pages_viewed * rng.uniform(0.3, 0.8, n)).astype(int), 0, 40)
    category_views = np.clip((product_views * rng.uniform(0.2, 0.6, n)).astype(int) + 1, 1, 8)

    # previous purchase history (loyalty signal)
    previous_purchases = rng.poisson(1.3, n)
    previous_purchases = np.clip(previous_purchases, 0, 25)
    days_since_last_purchase = np.where(
        previous_purchases > 0,
        rng.integers(1, 365, n),
        9999  # never purchased before
    )
    previous_order_value = np.where(
        previous_purchases > 0,
        np.round(np.clip(rng.normal(1800, 900, n), 300, 12000), 2),
        0.0
    )
    total_spent = np.round(previous_order_value * previous_purchases * rng.uniform(0.85, 1.15, n), 2)

    discount_used = rng.choice([0, 1], size=n, p=[0.62, 0.38])
    coupon_used = np.where(discount_used == 1, rng.choice([0, 1], size=n, p=[0.35, 0.65]), 0)
    wishlist_items = np.clip(rng.poisson(1.1, n), 0, 15)

    product_category = rng.choice(PRODUCT_CATEGORIES, size=n)
    product_price = np.round(np.clip(rng.normal(2200, 1100, n), 299, 15000), 2)

    # ---- Behavioural core: add_to_cart & checkout_started depend on engagement ----
    cart_propensity = sigmoid(
        -2.2
        + 0.12 * product_views
        + 0.07 * category_views
        + 0.25 * (session_duration / 10)
        + 0.18 * np.log1p(previous_purchases)
        + 0.15 * discount_used
        + rng.normal(0, 0.9, n)
    )
    add_to_cart = (rng.uniform(0, 1, n) < cart_propensity).astype(int) * rng.integers(1, 4, n)
    add_to_cart = np.where(rng.uniform(0, 1, n) < cart_propensity, add_to_cart, 0)

    cart_value = np.where(add_to_cart > 0,
                           np.round(product_price * add_to_cart * rng.uniform(0.9, 1.1, n), 2),
                           0.0)

    checkout_propensity = sigmoid(
        -1.5
        + 1.2 * (add_to_cart > 0).astype(int)
        + 0.08 * np.log1p(cart_value / 500)
        + 0.15 * np.log1p(previous_purchases)
        + 0.10 * coupon_used
        + rng.normal(0, 0.8, n)
    )
    checkout_started = (rng.uniform(0, 1, n) < checkout_propensity).astype(int)
    checkout_started = checkout_started * (add_to_cart > 0).astype(int)  # can't checkout w/o cart

    # ---- Final purchase probability driven by the funnel + engagement + loyalty ----
    # Modelled as a CONDITIONAL probability given funnel stage, to reflect real-world
    # cart/checkout abandonment (most e-commerce checkouts do NOT convert to a sale -
    # abandonment rates of 60-75% are typical), rather than treating checkout_started
    # as a near-deterministic proxy for purchase.
    traffic_boost = np.array([TRAFFIC_CONV_BOOST[t] for t in traffic_source])
    device_boost = np.where(device_type == "Mobile", 0.03,
                    np.where(device_type == "Desktop", 0.06, -0.02))
    loyalty_boost = 0.06 * np.log1p(previous_purchases)
    recency_penalty = 0.10 * np.minimum(days_since_last_purchase, 400) / 400
    engagement_boost = 0.05 * (session_duration / 10) + 0.01 * product_views
    misc_adjustment = traffic_boost + device_boost + loyalty_boost + engagement_boost \
        - recency_penalty + 0.05 * discount_used

    # Base conversion probability strongly depends on how far a session got in the
    # funnel, with meaningful noise added on top of each base rate.
    base_prob = np.select(
        [checkout_started == 1, add_to_cart > 0],
        [0.55, 0.16],
        default=0.02
    )
    purchase_prob_true = np.clip(base_prob + misc_adjustment + rng.normal(0, 0.07, n), 0.01, 0.96)
    purchase = (rng.uniform(0, 1, n) < purchase_prob_true).astype(int)

    purchase_value = np.where(
        purchase == 1,
        np.round(np.where(cart_value > 0, cart_value, product_price) * rng.uniform(0.95, 1.05, n), 2),
        0.0
    )

    # NOTE: customer_lifetime_value must be derived ONLY from historical behaviour
    # (total_spent, previous_purchases), never from the current session's outcome
    # (purchase / purchase_value) - otherwise it leaks the prediction target.
    customer_lifetime_value = np.round(
        total_spent * rng.uniform(1.05, 1.35, n)
        + previous_purchases * rng.uniform(150, 400, n)
        + rng.normal(0, 120, n),
        2
    )
    customer_lifetime_value = np.clip(customer_lifetime_value, 0, None)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "age": age,
        "gender": gender,
        "city": city,
        "device_type": device_type,
        "traffic_source": traffic_source,
        "session_duration": session_duration,
        "pages_viewed": pages_viewed,
        "product_views": product_views,
        "category_views": category_views,
        "add_to_cart": add_to_cart,
        "cart_value": cart_value,
        "checkout_started": checkout_started,
        "previous_purchases": previous_purchases,
        "previous_order_value": previous_order_value,
        "discount_used": discount_used,
        "coupon_used": coupon_used,
        "wishlist_items": wishlist_items,
        "product_category": product_category,
        "product_price": product_price,
        "days_since_last_purchase": days_since_last_purchase,
        "total_spent": total_spent,
        "customer_lifetime_value": customer_lifetime_value,
        "purchase": purchase,
        "purchase_value": purchase_value,
    })

    # ---- Inject realistic messiness: missing values, duplicates ----
    # Missing values in a few columns (simulate real-world tracking gaps)
    for col, frac in [("age", 0.02), ("city", 0.015), ("session_duration", 0.01),
                       ("previous_order_value", 0.01)]:
        mask = rng.uniform(0, 1, n) < frac
        df.loc[mask, col] = np.nan

    # A handful of duplicate rows (simulate double-fired tracking events)
    dup_frac = 0.008
    n_dupes = int(n * dup_frac)
    dupe_rows = df.sample(n=n_dupes, random_state=seed)
    df = pd.concat([df, dupe_rows], ignore_index=True)

    # A few extreme outliers in session_duration / cart_value (real users do weird things)
    outlier_idx = rng.choice(df.index, size=max(5, int(len(df) * 0.003)), replace=False)
    df.loc[outlier_idx, "session_duration"] = df.loc[outlier_idx, "session_duration"] * rng.uniform(8, 15)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def generate_product_catalogue(seed=RANDOM_SEED):
    """Generates a fictional jewelry product catalogue for the recommendation module."""
    rng = np.random.default_rng(seed + 1)

    products = [
        ("Everbond Couple Bracelet", "Bracelets", "Everbond", "couple, matching, adjustable, silver"),
        ("Infinity Charm Bracelet", "Bracelets", "Lumora", "charm, infinity, minimal, everyday"),
        ("Matching Heart Bracelet Set", "Bracelets", "Everbond", "couple, heart, set, gift"),
        ("Classic Chain Bracelet", "Bracelets", "Solstice", "chain, minimal, unisex, everyday"),
        ("Couple Initial Ring", "Rings", "Solstice", "couple, initial, engraved, personalized"),
        ("Promise Band Ring", "Rings", "Lumora", "promise, band, minimal, everyday"),
        ("Personalized Name Ring", "Rings", "Everbond", "personalized, engraved, gift, name"),
        ("Stackable Stone Ring", "Rings", "Aurelia", "stackable, stone, statement, party"),
        ("Twin Souls Pendant", "Pendants", "Everbond", "couple, pendant, heart, matching"),
        ("Zodiac Sign Pendant", "Pendants", "Aurelia", "zodiac, personalized, gift, minimal"),
        ("Layered Initial Pendant", "Pendants", "Lumora", "layered, initial, minimal, everyday"),
        ("Couple Pendant Set", "Pendants", "Everbond", "couple, set, matching, gift"),
        ("Minimalist Chain Necklace", "Necklaces", "Solstice", "minimal, chain, everyday, unisex"),
        ("Statement Layered Necklace", "Necklaces", "Aurelia", "statement, layered, party, bold"),
        ("Couple Matching Necklace Set", "Necklaces", "Everbond", "couple, matching, set, gift"),
        ("Pearl Drop Necklace", "Necklaces", "Lumora", "pearl, drop, elegant, evening"),
        ("Stud Earrings Classic", "Earrings", "Solstice", "stud, minimal, everyday, classic"),
        ("Hoop Earrings Bold", "Earrings", "Aurelia", "hoop, statement, party, bold"),
        ("Couple Symbol Earrings", "Earrings", "Everbond", "couple, symbol, matching, gift"),
        ("Drop Earrings Elegant", "Earrings", "Lumora", "drop, elegant, evening, party"),
        ("Chain Anklet Minimal", "Anklets", "Solstice", "anklet, minimal, everyday, chain"),
        ("Charm Anklet Beaded", "Anklets", "Lumora", "anklet, beaded, boho, everyday"),
        ("Tiny Stud Nose Pin", "Nose Pins", "Solstice", "nose pin, minimal, everyday, tiny"),
        ("Floral Nose Pin", "Nose Pins", "Aurelia", "nose pin, floral, traditional, statement"),
        ("His & Hers Couple Set", "Couple Sets", "Everbond", "couple, set, matching, anniversary"),
        ("Anniversary Couple Set", "Couple Sets", "Everbond", "couple, anniversary, gift, matching"),
        ("Everyday Couple Bands", "Couple Sets", "Solstice", "couple, band, everyday, minimal"),
        ("Celestial Couple Pendant Set", "Couple Sets", "Aurelia", "couple, celestial, gift, matching"),
    ]

    rows = []
    for i, (name, cat, brand, tags) in enumerate(products):
        price = round(float(rng.uniform(499, 4999)), 2)
        rows.append({
            "product_id": f"PROD{1000+i}",
            "product_name": name,
            "category": cat,
            "price": price,
            "brand": brand,
            "tags": tags,
            "description": f"{name} - a {cat.lower()} piece by {brand}, crafted in 925 sterling silver, "
                            f"tagged as {tags}."
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating e-commerce customer dataset...")
    df = generate_dataset()
    out_path = os.path.join(OUTPUT_DIR, "ecommerce_customer_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} records -> {out_path}")
    print(f"Purchase rate: {df['purchase'].mean():.2%}")
    print(f"Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")

    print("\nGenerating product catalogue...")
    products = generate_product_catalogue()
    prod_path = os.path.join(OUTPUT_DIR, "products.csv")
    products.to_csv(prod_path, index=False)
    print(f"Saved {len(products)} products -> {prod_path}")
