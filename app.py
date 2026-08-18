"""
app.py
-------
Streamlit dashboard for the E-Commerce Customer Intelligence & Purchase
Prediction System. Run with:

    streamlit run app.py

Pages:
  - Dashboard (KPIs, funnel, revenue, device/traffic/category performance)
  - Customer Segmentation (K-Means clusters, filters)
  - Purchase Prediction (interactive form + probability + top factors)
  - Product Recommendations (content-based TF-IDF recommender)
  - Customer Risk (RFM-style risk scoring, filters, CSV download)
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.utils import (
    DATASET_PATH, PRODUCTS_PATH, PURCHASE_MODEL_PATH, PREPROCESSING_PATH,
    CLUSTERING_MODEL_PATH, CLUSTER_SCALER_PATH, FEATURE_LIST_PATH, METRICS_PATH,
    format_currency, format_percent
)
from src.data_preprocessing import (
    full_preprocessing_pipeline, build_feature_matrix, align_features,
    CATEGORICAL_COLS
)
from src.eda import (
    conversion_summary, purchase_distribution_fig, conversion_by_dimension_fig,
    revenue_by_category_fig, session_duration_vs_purchase_fig,
    product_views_vs_purchase_fig, funnel_fig, revenue_trend_fig,
    clv_distribution_fig, add_to_cart_vs_purchase_fig, previous_purchases_vs_future_fig
)
from src.segmentation import run_segmentation, SEGMENTATION_FEATURES
from src.churn_analysis import compute_rfm, risk_summary
from src.recommendation import fit_tfidf, recommend_products

st.set_page_config(page_title="E-Commerce Customer Intelligence", layout="wide",
                    page_icon="\U0001F6CD\uFE0F")


# ------------------------------------------------------------------
# Cached data / model loaders
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_data():
    df = full_preprocessing_pipeline(DATASET_PATH)
    return df


@st.cache_data(show_spinner="Loading product catalogue...")
def load_products():
    return pd.read_csv(PRODUCTS_PATH)


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    purchase_model = joblib.load(PURCHASE_MODEL_PATH)
    scaler = joblib.load(PREPROCESSING_PATH)
    cluster_model = joblib.load(CLUSTERING_MODEL_PATH)
    cluster_scaler = joblib.load(CLUSTER_SCALER_PATH)
    with open(FEATURE_LIST_PATH) as f:
        feature_info = json.load(f)
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}
    return purchase_model, scaler, cluster_model, cluster_scaler, feature_info, metrics


@st.cache_resource(show_spinner="Preparing recommendation engine...")
def load_recommender(products_df):
    return fit_tfidf(products_df)


@st.cache_data(show_spinner="Segmenting customers...")
def get_customer_segments(df):
    customer_df, _, _, summary = run_segmentation(df, n_clusters=4)
    return customer_df, summary


@st.cache_data(show_spinner="Scoring customer risk...")
def get_risk_scores(customer_df):
    rfm = compute_rfm(customer_df)
    return rfm


df = load_data()
products_df = load_products()

try:
    purchase_model, scaler, cluster_model, cluster_scaler, feature_info, metrics = load_models()
    MODELS_LOADED = True
except FileNotFoundError:
    MODELS_LOADED = False

vectorizer, tfidf_matrix, products_corpus_df = load_recommender(products_df)
customer_df, cluster_summary = get_customer_segments(df)
rfm_df = get_risk_scores(customer_df)


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("\U0001F6CD\uFE0F E-Commerce Intelligence")
st.sidebar.caption("Customer Behaviour \u2022 ML Predictions \u2022 Business Insights")
page = st.sidebar.radio("Navigate", [
    "\U0001F4CA Dashboard",
    "\U0001F465 Customer Segmentation",
    "\U0001F3AF Purchase Prediction",
    "\U0001F48D Product Recommendations",
    "\u26A0\uFE0F Customer Risk",
])
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} sessions \u2022 {df['customer_id'].nunique():,} customers")
if not MODELS_LOADED:
    st.sidebar.warning("Models not found. Run `python train_models.py` first.")

if not MODELS_LOADED:
    st.error(
        "Trained models were not found in the `models/` folder.\n\n"
        "Please run the following before launching the dashboard:\n\n"
        "```\npython generate_dataset.py\npython setup_database.py\npython train_models.py\n```"
    )
    st.stop()


# ------------------------------------------------------------------
# PAGE 1: DASHBOARD
# ------------------------------------------------------------------
if page == "\U0001F4CA Dashboard":
    st.title("Business Intelligence Dashboard")
    st.caption("Overview of customer behaviour, conversion, and revenue performance")

    kpis = conversion_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{kpis['total_customers']:,}")
    c2.metric("Total Sessions", f"{kpis['total_sessions']:,}")
    c3.metric("Total Orders", f"{kpis['total_orders']:,}")
    c4.metric("Conversion Rate", format_percent(kpis['conversion_rate']))

    c5, c6, c7 = st.columns(3)
    c5.metric("Total Revenue", format_currency(kpis['revenue']))
    c6.metric("Avg Order Value", format_currency(kpis['avg_order_value']))
    c7.metric("Avg Customer LTV", format_currency(kpis['avg_clv']))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(funnel_fig(df), use_container_width=True)
    with col2:
        st.plotly_chart(purchase_distribution_fig(df), use_container_width=True)

    st.plotly_chart(revenue_trend_fig(df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig, _ = conversion_by_dimension_fig(df, "device_type", "Conversion Rate by Device")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig, _ = conversion_by_dimension_fig(df, "traffic_source", "Conversion Rate by Traffic Source")
        st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(revenue_by_category_fig(df), use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(add_to_cart_vs_purchase_fig(df), use_container_width=True)
    with col6:
        st.plotly_chart(previous_purchases_vs_future_fig(df), use_container_width=True)

    with st.expander("More exploratory analysis: session duration, product views, CLV"):
        st.plotly_chart(session_duration_vs_purchase_fig(df), use_container_width=True)
        st.plotly_chart(product_views_vs_purchase_fig(df), use_container_width=True)
        st.plotly_chart(clv_distribution_fig(df), use_container_width=True)


# ------------------------------------------------------------------
# PAGE 2: CUSTOMER SEGMENTATION
# ------------------------------------------------------------------
elif page == "\U0001F465 Customer Segmentation":
    st.title("Customer Segmentation")
    st.caption("K-Means clustering on behavioural & monetary features, "
               "with automatically-named segments based on cluster characteristics")

    st.subheader("Segment Overview")
    display_summary = cluster_summary[[
        "segment_name", "size", "avg_spent", "avg_purchases", "avg_clv",
        "avg_recency", "avg_engagement", "avg_purchase_rate"
    ]].rename(columns={
        "segment_name": "Segment", "size": "Customers", "avg_spent": "Avg Spent (₹)",
        "avg_purchases": "Avg Past Purchases", "avg_clv": "Avg CLV (₹)",
        "avg_recency": "Avg Days Since Last Purchase", "avg_engagement": "Avg Product Views",
        "avg_purchase_rate": "Purchase Rate"
    })
    st.dataframe(display_summary.style.format({
        "Avg Spent (₹)": "{:,.0f}", "Avg Past Purchases": "{:.1f}",
        "Avg CLV (₹)": "{:,.0f}", "Avg Days Since Last Purchase": "{:.0f}",
        "Avg Product Views": "{:.1f}", "Purchase Rate": "{:.1%}"
    }), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        import plotly.express as px
        fig = px.pie(cluster_summary, values="size", names="segment_name",
                     title="Customer Distribution by Segment", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(cluster_summary.sort_values("avg_spent"), x="segment_name", y="avg_spent",
                      title="Average Spend by Segment", color="segment_name")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Customer-Level Segment Table")
    seg_filter = st.multiselect("Filter by segment", options=cluster_summary["segment_name"].tolist(),
                                 default=cluster_summary["segment_name"].tolist())
    filtered = customer_df[customer_df["segment"].isin(seg_filter)]
    st.dataframe(
        filtered[["customer_id", "segment", "total_spent", "previous_purchases",
                  "customer_lifetime_value", "product_views", "sessions"]].head(500),
        use_container_width=True
    )
    st.caption(f"Showing up to 500 of {len(filtered):,} filtered customers.")


# ------------------------------------------------------------------
# PAGE 3: PURCHASE PREDICTION
# ------------------------------------------------------------------
elif page == "\U0001F3AF Purchase Prediction":
    st.title("Purchase Prediction")
    st.caption(f"Model in use: **{feature_info.get('best_model', 'N/A')}** "
               "(selected by highest ROC-AUC on held-out test data)")

    with st.form("prediction_form"):
        st.subheader("Customer / Session Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=13, max_value=90, value=28)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            city = st.selectbox("City", sorted(df["city"].unique()))
            device_type = st.selectbox("Device", ["Mobile", "Desktop", "Tablet"])
        with c2:
            traffic_source = st.selectbox("Traffic Source",
                ["Organic Search", "Paid Ads", "Social Media", "Direct", "Email", "Referral"])
            session_duration = st.slider("Session Duration (minutes)", 0.0, 60.0, 8.0)
            pages_viewed = st.slider("Pages Viewed", 1, 40, 6)
            product_views = st.slider("Product Views", 0, 30, 7)
        with c3:
            category_views = st.slider("Category Views", 1, 8, 2)
            add_to_cart = st.slider("Items Added to Cart", 0, 5, 2)
            checkout_started = st.selectbox("Checkout Started?", ["Yes", "No"])
            previous_purchases = st.slider("Previous Purchases", 0, 20, 3)

        c4, c5, c6 = st.columns(3)
        with c4:
            discount_used = st.selectbox("Discount Used?", ["Yes", "No"])
            coupon_used = st.selectbox("Coupon Used?", ["Yes", "No"])
        with c5:
            wishlist_items = st.slider("Wishlist Items", 0, 10, 1)
            product_category = st.selectbox("Product Category", sorted(df["product_category"].unique()))
        with c6:
            product_price = st.number_input("Product Price (₹)", min_value=99.0, value=2000.0, step=50.0)
            days_since_last_purchase = st.slider("Days Since Last Purchase", 0, 400, 60)

        submitted = st.form_submit_button("Predict Purchase Probability", type="primary")

    if submitted:
        cart_value = product_price * add_to_cart if add_to_cart > 0 else 0.0
        previous_order_value = (product_price * 0.9) if previous_purchases > 0 else 0.0
        total_spent = previous_order_value * previous_purchases
        avg_order_value = total_spent / previous_purchases if previous_purchases > 0 else 0.0
        customer_lifetime_value = total_spent * 1.15 + previous_purchases * 250
        engagement_score = (0.3 * product_views + 0.2 * category_views + 0.3 * session_duration
                             + 1.5 * add_to_cart + 2.0 * (1 if checkout_started == "Yes" else 0))
        cart_conversion_ratio = (add_to_cart / product_views) if product_views > 0 else 0.0

        row = pd.DataFrame([{
            "age": age, "gender": gender, "city": city, "device_type": device_type,
            "traffic_source": traffic_source, "session_duration": session_duration,
            "pages_viewed": pages_viewed, "product_views": product_views,
            "category_views": category_views, "add_to_cart": add_to_cart,
            "cart_value": cart_value, "checkout_started": 1 if checkout_started == "Yes" else 0,
            "previous_purchases": previous_purchases, "previous_order_value": previous_order_value,
            "discount_used": 1 if discount_used == "Yes" else 0,
            "coupon_used": 1 if coupon_used == "Yes" else 0,
            "wishlist_items": wishlist_items, "product_category": product_category,
            "product_price": product_price, "days_since_last_purchase": days_since_last_purchase,
            "total_spent": total_spent, "customer_lifetime_value": customer_lifetime_value,
            "avg_order_value": avg_order_value, "engagement_score": engagement_score,
            "cart_conversion_ratio": cart_conversion_ratio,
            "days_since_last_purchase_capped": min(days_since_last_purchase, 400),
            "is_returning_customer": 1 if previous_purchases > 0 else 0,
        }])

        X_row, _ = build_feature_matrix(row)
        X_row = align_features(X_row, feature_info["feature_names"])
        numeric_cols = feature_info["numeric_cols"]
        X_row_scaled = X_row.copy()
        X_row_scaled[numeric_cols] = scaler.transform(X_row[numeric_cols])

        proba = purchase_model.predict_proba(X_row_scaled)[0, 1]

        from src.purchase_prediction import purchase_intent_label, explain_prediction
        label = purchase_intent_label(proba)

        st.markdown("---")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Purchase Probability", f"{proba*100:.1f}%")
            if label == "HIGH PURCHASE INTENT":
                st.success(f"**{label}**")
            elif label == "MEDIUM PURCHASE INTENT":
                st.warning(f"**{label}**")
            else:
                st.error(f"**{label}**")
        with col_b:
            st.subheader("Top Contributing Factors")
            top_feats = explain_prediction(purchase_model, feature_info["feature_names"], top_n=6)
            factors_df = pd.DataFrame(top_feats, columns=["Feature", "Model Importance"])
            st.dataframe(factors_df, use_container_width=True, hide_index=True)
            st.caption(
                "Importance reflects each feature's overall influence on the trained model's "
                "decisions (global feature importance) - a lightweight, dependency-free "
                "alternative to SHAP used here for interpretability."
            )


# ------------------------------------------------------------------
# PAGE 4: PRODUCT RECOMMENDATIONS
# ------------------------------------------------------------------
elif page == "\U0001F48D Product Recommendations":
    st.title("Product Recommendations")
    st.caption("Content-based recommendations using TF-IDF + cosine similarity over "
               "category, brand, tags, and description")

    categories = sorted(products_df["category"].unique())
    viewed_categories = st.multiselect("Categories viewed by the customer", categories,
                                        default=categories[:2])
    viewed_products = st.multiselect("Specific products viewed (optional)",
                                      products_df["product_name"].tolist())
    top_n = st.slider("Number of recommendations", 3, 10, 5)

    if st.button("Get Recommendations", type="primary"):
        recs = recommend_products(viewed_categories, viewed_products, products_df,
                                    vectorizer, tfidf_matrix, top_n=top_n)
        st.subheader(f"Top {len(recs)} Recommended Products")
        recs_display = recs.rename(columns={
            "product_name": "Product", "category": "Category", "price": "Price (₹)",
            "similarity_score": "Similarity Score"
        })
        st.dataframe(recs_display.style.format({"Price (₹)": "{:,.0f}", "Similarity Score": "{:.3f}"}),
                     use_container_width=True, hide_index=True)


# ------------------------------------------------------------------
# PAGE 5: CUSTOMER RISK
# ------------------------------------------------------------------
elif page == "\u26A0\uFE0F Customer Risk":
    st.title("Customer Risk / Churn Analysis")
    st.caption(
        "This is a business RFM-style risk score (Recency, Frequency, Monetary, Engagement) "
        "- a transparent scoring system, not a medical/financial prediction and distinct "
        "from the supervised purchase-prediction model."
    )

    summary = risk_summary(rfm_df)
    c1, c2, c3 = st.columns(3)
    for col, cat in zip([c1, c2, c3], ["Low Risk", "Medium Risk", "High Risk"]):
        row = summary[summary["risk_category"] == cat]
        count = int(row["customers"].iloc[0]) if len(row) else 0
        col.metric(cat, f"{count:,} customers")

    risk_filter = st.multiselect("Filter by risk category", ["Low Risk", "Medium Risk", "High Risk"],
                                  default=["High Risk", "Medium Risk", "Low Risk"])
    filtered = rfm_df[rfm_df["risk_category"].isin(risk_filter)]

    display_cols = ["customer_id", "risk_category", "risk_score", "total_spent",
                     "previous_purchases", "days_since_last_purchase_capped"]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "risk_category": "Risk Category", "risk_score": "Risk Score (0-100)",
            "total_spent": "Total Spent (₹)", "previous_purchases": "Previous Purchases",
            "days_since_last_purchase_capped": "Days Since Last Purchase"
        }).sort_values("Risk Score (0-100)").head(500),
        use_container_width=True, hide_index=True
    )
    st.caption(f"Showing up to 500 of {len(filtered):,} filtered customers. "
               "Higher score = healthier / lower risk.")

    csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered customer risk data (CSV)", csv,
                        "customer_risk_scores.csv", "text/csv")
