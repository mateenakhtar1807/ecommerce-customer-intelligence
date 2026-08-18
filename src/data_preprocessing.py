"""
src/data_preprocessing.py
---------------------------
Handles data cleaning and preprocessing for the e-commerce dataset:
missing values, duplicates, outliers, data types, encoding, and scaling.

Design notes / reasoning:
- Missing numeric values (age, session_duration, previous_order_value) are
  filled with the MEDIAN rather than mean, because these fields are
  right-skewed (a few very high spenders / long sessions) and the median is
  more robust to that skew.
- Missing categorical values (city) are filled with the MODE, which is the
  standard, interpretable approach for low-cardinality categorical data.
- Duplicates are dropped based on session_id, since a duplicated session_id
  represents a double-fired tracking event, not a genuine second session.
- Outliers are NOT blindly removed. Instead, extreme values in
  session_duration and cart_value are CAPPED (winsorized) at the 99th
  percentile. This is preferred over deletion because purchase behaviour
  from genuinely high-engagement users is exactly the signal the purchase
  prediction model needs - deleting those rows would throw away real signal.
  Capping keeps the row but prevents a handful of extreme values from
  destabilising models that are sensitive to scale (Logistic Regression,
  K-Means).
"""

import pandas as pd
import numpy as np


NUMERIC_MEDIAN_FILL_COLS = ["age", "session_duration", "previous_order_value"]
CATEGORICAL_MODE_FILL_COLS = ["city"]

OUTLIER_CAP_COLS = ["session_duration", "cart_value", "purchase_value", "total_spent"]

CATEGORICAL_COLS = ["gender", "city", "device_type", "traffic_source", "product_category"]

NUMERIC_FEATURE_COLS = [
    "age", "session_duration", "pages_viewed", "product_views", "category_views",
    "add_to_cart", "cart_value", "checkout_started", "previous_purchases",
    "previous_order_value", "discount_used", "coupon_used", "wishlist_items",
    "product_price", "days_since_last_purchase", "total_spent",
    "customer_lifetime_value",
]


def load_raw_data(path):
    """Load the raw CSV dataset and parse the timestamp column."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pass: duplicates -> missing values -> outlier capping ->
    dtype fixes -> sanity constraints. Returns a new, cleaned DataFrame.
    """
    df = df.copy()

    # 1. Remove duplicate sessions (double-fired tracking events)
    before = len(df)
    df = df.drop_duplicates(subset=["session_id"], keep="first")
    removed = before - len(df)

    # 2. Missing value handling
    for col in NUMERIC_MEDIAN_FILL_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_MODE_FILL_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Fix data types
    df["age"] = df["age"].round().astype(int)
    for col in ["add_to_cart", "checkout_started", "previous_purchases",
                "discount_used", "coupon_used", "wishlist_items", "purchase"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # 4. Outlier capping (winsorize at 1st / 99th percentile) - not deletion
    for col in OUTLIER_CAP_COLS:
        if col in df.columns:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)

    # 5. Sanity constraints (business logic guardrails)
    df["age"] = df["age"].clip(13, 90)
    df["checkout_started"] = np.where(df["add_to_cart"] == 0, 0, df["checkout_started"])

    df = df.reset_index(drop=True)
    df.attrs["duplicates_removed"] = removed
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features useful for both the purchase-prediction model and
    the customer-segmentation model.
    """
    df = df.copy()

    # Average order value per customer-history (avoids div-by-zero)
    df["avg_order_value"] = np.where(
        df["previous_purchases"] > 0,
        df["total_spent"] / df["previous_purchases"].replace(0, np.nan),
        0
    )
    df["avg_order_value"] = df["avg_order_value"].fillna(0)

    # Engagement score: simple weighted combination of funnel signals
    df["engagement_score"] = (
        0.3 * df["product_views"]
        + 0.2 * df["category_views"]
        + 0.3 * df["session_duration"]
        + 1.5 * df["add_to_cart"]
        + 2.0 * df["checkout_started"]
    )

    # Cart-to-view ratio - proxy for purchase intent
    df["cart_conversion_ratio"] = np.where(
        df["product_views"] > 0, df["add_to_cart"] / df["product_views"], 0
    )

    # Recency bucket: cap "never purchased" (9999) into a large-but-finite value for modelling
    df["days_since_last_purchase_capped"] = df["days_since_last_purchase"].clip(upper=400)

    # Is a returning customer
    df["is_returning_customer"] = (df["previous_purchases"] > 0).astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame, categorical_cols=None, numeric_cols=None):
    """
    One-hot encodes categorical columns and returns (X, feature_names).
    Used identically at training time and inference time to guarantee
    consistent columns.
    """
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    numeric_cols = numeric_cols or (NUMERIC_FEATURE_COLS + [
        "avg_order_value", "engagement_score", "cart_conversion_ratio",
        "days_since_last_purchase_capped", "is_returning_customer"
    ])

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    X_numeric = df[numeric_cols].copy()
    X_categorical = pd.get_dummies(df[categorical_cols], drop_first=False)
    X = pd.concat([X_numeric, X_categorical], axis=1)
    return X, list(X.columns)


def align_features(X: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """
    Ensures a feature matrix has exactly `expected_columns`, in that order,
    filling any missing dummy columns with 0. Critical for inference time,
    where a single customer's one-hot encoding won't produce every category.
    """
    X = X.copy()
    for col in expected_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_columns]
    return X


def full_preprocessing_pipeline(raw_csv_path):
    """Convenience wrapper: load -> clean -> engineer features."""
    df = load_raw_data(raw_csv_path)
    df = clean_data(df)
    df = engineer_features(df)
    return df
