"""
src/segmentation.py
---------------------
Unsupervised customer segmentation using K-Means clustering.

Approach:
1. Aggregate the session-level data to CUSTOMER level (each customer may
   have multiple sessions in the raw data).
2. Scale the chosen behavioural/monetary features with StandardScaler
   (K-Means is distance-based, so unscaled features like total_spent in
   rupees would dominate over ratios like cart_conversion_ratio).
3. Use the Elbow Method (inertia) and Silhouette Score to select a
   reasonable K (we search K=2..8 and report both diagnostics).
4. Fit final K-Means model.
5. Interpret clusters by their average feature values, then map cluster
   IDs to human-readable segment names algorithmically (NOT hardcoded to a
   specific cluster number) - ranked by a composite value score.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

SEGMENTATION_FEATURES = [
    "total_spent", "previous_purchases", "avg_order_value",
    "product_views", "add_to_cart", "session_duration",
    "days_since_last_purchase_capped", "customer_lifetime_value",
]


def build_customer_level_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate session-level rows to one row per customer."""
    agg = df.groupby("customer_id").agg(
        total_spent=("total_spent", "max"),
        previous_purchases=("previous_purchases", "max"),
        avg_order_value=("avg_order_value", "max"),
        product_views=("product_views", "mean"),
        add_to_cart=("add_to_cart", "mean"),
        session_duration=("session_duration", "mean"),
        days_since_last_purchase_capped=("days_since_last_purchase_capped", "min"),
        customer_lifetime_value=("customer_lifetime_value", "max"),
        sessions=("session_id", "nunique"),
        purchase_rate=("purchase", "mean"),
    ).reset_index()
    return agg


def find_optimal_k(X_scaled, k_range=range(2, 9), random_state=42):
    """Returns a DataFrame of inertia + silhouette score for each K (Elbow + Silhouette)."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else np.nan
        results.append({"k": k, "inertia": km.inertia_, "silhouette_score": sil})
    return pd.DataFrame(results)


def fit_kmeans(X_scaled, n_clusters=4, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def name_clusters(customer_df: pd.DataFrame, cluster_col="cluster") -> dict:
    """
    Analyses cluster characteristics and algorithmically assigns meaningful
    business names. We compute a composite "value score" per cluster from
    z-scored spend/engagement/recency, then rank clusters by that score and
    assign names in order - so names are never tied to a hardcoded cluster
    ID and adapt automatically if K-Means labels shuffle between runs.
    """
    summary = customer_df.groupby(cluster_col).agg(
        avg_spent=("total_spent", "mean"),
        avg_purchases=("previous_purchases", "mean"),
        avg_clv=("customer_lifetime_value", "mean"),
        avg_recency=("days_since_last_purchase_capped", "mean"),
        avg_engagement=("product_views", "mean"),
        avg_purchase_rate=("purchase_rate", "mean"),
        size=(cluster_col, "count"),
    ).reset_index()

    # Composite value score: high spend/purchases/CLV/engagement are good,
    # high recency (days since last purchase) is bad -> subtract it.
    def z(s):
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0

    summary["value_score"] = (
        z(summary["avg_spent"]) + z(summary["avg_purchases"]) +
        z(summary["avg_clv"]) + z(summary["avg_engagement"]) +
        z(summary["avg_purchase_rate"]) - z(summary["avg_recency"])
    )
    summary = summary.sort_values("value_score", ascending=False).reset_index(drop=True)

    n = len(summary)
    if n == 4:
        names = ["High Value Customers", "Loyal Customers", "Potential Customers", "At-Risk Customers"]
    elif n == 3:
        names = ["High Value Customers", "Potential Customers", "At-Risk Customers"]
    elif n == 2:
        names = ["High Value Customers", "At-Risk Customers"]
    else:
        # Generic fallback for other K: top/bottom get clear labels, middle ranked
        names = []
        for i in range(n):
            if i == 0:
                names.append("High Value Customers")
            elif i == n - 1:
                names.append("At-Risk Customers")
            else:
                names.append(f"Growth Segment {i}")

    summary["segment_name"] = names
    cluster_to_name = dict(zip(summary[cluster_col], summary["segment_name"]))
    return cluster_to_name, summary


def run_segmentation(df: pd.DataFrame, n_clusters=4, random_state=42):
    """Full pipeline: aggregate -> scale -> cluster -> name. Returns customer_df, model, scaler, summary."""
    customer_df = build_customer_level_data(df)
    X = customer_df[SEGMENTATION_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model, labels = fit_kmeans(X_scaled, n_clusters=n_clusters, random_state=random_state)
    customer_df["cluster"] = labels

    cluster_to_name, summary = name_clusters(customer_df)
    customer_df["segment"] = customer_df["cluster"].map(cluster_to_name)
    summary["segment_name"] = summary["segment_name"]

    return customer_df, model, scaler, summary


def predict_segment(model, scaler, cluster_to_name, feature_row: pd.DataFrame):
    """Predict the segment for a single new customer-level feature row."""
    X = feature_row[SEGMENTATION_FEATURES].fillna(0)
    X_scaled = scaler.transform(X)
    cluster = model.predict(X_scaled)[0]
    return cluster_to_name.get(cluster, f"Cluster {cluster}")
