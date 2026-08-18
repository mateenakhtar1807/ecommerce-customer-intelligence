"""
src/churn_analysis.py
-----------------------
Business-style Customer Risk Scoring using an RFM (Recency, Frequency,
Monetary) approach plus engagement.

IMPORTANT DISTINCTION: this is a rule-based, interpretable BUSINESS SCORING
module - not a supervised ML classifier, and not a medical/financial risk
model. It quantiles customers on four dimensions, combines them into a
0-100 risk score, and buckets customers as Low / Medium / High risk. This
is the same style of scoring real e-commerce and CRM teams use (e.g. RFM
segmentation) because it's transparent and easy to explain to a
non-technical stakeholder, unlike a black-box classifier.
"""

import pandas as pd
import numpy as np


def compute_rfm(customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a customer-level DataFrame with:
    days_since_last_purchase_capped, previous_purchases, total_spent,
    product_views / session_duration (engagement).
    Returns the same df with R, F, M, E quantile scores (1-5, 5=best) and a
    combined risk_score (0-100, higher = healthier / lower risk) plus a
    risk_category.
    """
    df = customer_df.copy()

    # Recency: LOWER days-since-last-purchase is better -> invert quantile
    df["R_score"] = pd.qcut(df["days_since_last_purchase_capped"].rank(method="first"),
                             5, labels=[5, 4, 3, 2, 1]).astype(int)
    # Frequency: more previous purchases is better
    df["F_score"] = pd.qcut(df["previous_purchases"].rank(method="first"),
                             5, labels=[1, 2, 3, 4, 5]).astype(int)
    # Monetary: higher total spend is better
    df["M_score"] = pd.qcut(df["total_spent"].rank(method="first"),
                             5, labels=[1, 2, 3, 4, 5]).astype(int)
    # Engagement: higher product_views/session_duration is better
    engagement_raw = df["product_views"].fillna(0) + df["session_duration"].fillna(0)
    df["E_score"] = pd.qcut(engagement_raw.rank(method="first"),
                             5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Weighted composite -> 0-100 scale (Recency & Frequency weighted highest,
    # since a customer who hasn't purchased recently AND rarely buys is the
    # clearest churn signal)
    df["risk_score"] = (
        (df["R_score"] * 0.35 + df["F_score"] * 0.30 +
         df["M_score"] * 0.20 + df["E_score"] * 0.15) / 5 * 100
    ).round(1)

    df["risk_category"] = pd.cut(
        df["risk_score"], bins=[-1, 40, 65, 101],
        labels=["High Risk", "Medium Risk", "Low Risk"]
    )
    return df


def risk_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    summary = rfm_df.groupby("risk_category", observed=True).agg(
        customers=("customer_id", "count"),
        avg_spent=("total_spent", "mean"),
        avg_recency_days=("days_since_last_purchase_capped", "mean"),
        avg_risk_score=("risk_score", "mean"),
    ).reset_index()
    return summary
