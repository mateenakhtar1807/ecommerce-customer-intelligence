"""
src/eda.py
-----------
Reusable Exploratory Data Analysis functions. Each function returns either a
summary DataFrame or a Plotly figure, so they can be called both from the
Streamlit dashboard and from train_models.py (for saving static reports).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def conversion_summary(df: pd.DataFrame) -> dict:
    """High level KPI numbers for the dashboard header."""
    total_sessions = len(df)
    total_customers = df["customer_id"].nunique()
    total_orders = int(df["purchase"].sum())
    conversion_rate = df["purchase"].mean()
    revenue = df["purchase_value"].sum()
    aov = df.loc[df["purchase"] == 1, "purchase_value"].mean() if total_orders > 0 else 0
    avg_clv = df["customer_lifetime_value"].mean()
    return {
        "total_sessions": total_sessions,
        "total_customers": total_customers,
        "total_orders": total_orders,
        "conversion_rate": conversion_rate,
        "revenue": revenue,
        "avg_order_value": aov,
        "avg_clv": avg_clv,
    }


def purchase_distribution_fig(df: pd.DataFrame):
    counts = df["purchase"].map({0: "Did Not Purchase", 1: "Purchased"}).value_counts()
    fig = px.pie(values=counts.values, names=counts.index, hole=0.45,
                 title="Purchase vs Non-Purchase",
                 color_discrete_sequence=["#C9A66B", "#2E2E2E"])
    return fig


def conversion_by_dimension_fig(df: pd.DataFrame, dimension: str, title: str = None):
    """Bar chart of conversion rate grouped by a categorical dimension."""
    grp = df.groupby(dimension)["purchase"].agg(["mean", "count"]).reset_index()
    grp.columns = [dimension, "conversion_rate", "sessions"]
    grp = grp.sort_values("conversion_rate", ascending=False)
    fig = px.bar(grp, x=dimension, y="conversion_rate", text_auto=".1%",
                 title=title or f"Conversion Rate by {dimension.replace('_', ' ').title()}",
                 color="conversion_rate", color_continuous_scale="Sunset")
    fig.update_layout(yaxis_tickformat=".0%")
    return fig, grp


def revenue_by_category_fig(df: pd.DataFrame):
    grp = df[df["purchase"] == 1].groupby("product_category")["purchase_value"].sum().reset_index()
    grp = grp.sort_values("purchase_value", ascending=False)
    fig = px.bar(grp, x="product_category", y="purchase_value",
                 title="Revenue by Product Category", color="purchase_value",
                 color_continuous_scale="Sunset")
    return fig


def session_duration_vs_purchase_fig(df: pd.DataFrame):
    fig = px.box(df, x="purchase", y="session_duration", color="purchase",
                 labels={"purchase": "Purchased (0=No, 1=Yes)"},
                 title="Session Duration vs Purchase")
    return fig


def product_views_vs_purchase_fig(df: pd.DataFrame):
    grp = df.groupby("product_views")["purchase"].mean().reset_index()
    fig = px.line(grp, x="product_views", y="purchase", markers=True,
                   title="Product Views vs Purchase Probability")
    fig.update_layout(yaxis_tickformat=".0%")
    return fig


def funnel_fig(df: pd.DataFrame):
    stages = ["Sessions", "Product Views > 0", "Add to Cart", "Checkout Started", "Purchased"]
    values = [
        len(df),
        (df["product_views"] > 0).sum(),
        (df["add_to_cart"] > 0).sum(),
        (df["checkout_started"] > 0).sum(),
        (df["purchase"] > 0).sum(),
    ]
    fig = go.Figure(go.Funnel(y=stages, x=values, textinfo="value+percent initial",
                               marker={"color": ["#F5EFE6", "#E8D9C5", "#D6B98C", "#C9A66B", "#8C6B3F"]}))
    fig.update_layout(title="Conversion Funnel")
    return fig


def revenue_trend_fig(df: pd.DataFrame):
    purchases = df[df["purchase"] == 1].copy()
    purchases["date"] = pd.to_datetime(purchases["timestamp"]).dt.date
    trend = purchases.groupby("date")["purchase_value"].sum().reset_index()
    trend = trend.sort_values("date")
    fig = px.line(trend, x="date", y="purchase_value", title="Daily Revenue Trend")
    return fig


def clv_distribution_fig(df: pd.DataFrame):
    fig = px.histogram(df, x="customer_lifetime_value", nbins=50,
                        title="Customer Lifetime Value Distribution",
                        color_discrete_sequence=["#C9A66B"])
    return fig


def add_to_cart_vs_purchase_fig(df: pd.DataFrame):
    grp = df.copy()
    grp["cart_bucket"] = np.where(grp["add_to_cart"] == 0, "0", np.where(grp["add_to_cart"] == 1, "1", "2+"))
    grp = grp.groupby("cart_bucket")["purchase"].mean().reset_index()
    fig = px.bar(grp, x="cart_bucket", y="purchase", title="Add-to-Cart Count vs Purchase Rate",
                 text_auto=".1%")
    fig.update_layout(yaxis_tickformat=".0%")
    return fig


def previous_purchases_vs_future_fig(df: pd.DataFrame):
    grp = df.copy()
    grp["loyalty_bucket"] = pd.cut(grp["previous_purchases"], bins=[-1, 0, 2, 5, 100],
                                    labels=["0 (New)", "1-2", "3-5", "6+"])
    grp = grp.groupby("loyalty_bucket", observed=True)["purchase"].mean().reset_index()
    fig = px.bar(grp, x="loyalty_bucket", y="purchase",
                 title="Previous Purchases vs Future Purchase Rate", text_auto=".1%")
    fig.update_layout(yaxis_tickformat=".0%")
    return fig
