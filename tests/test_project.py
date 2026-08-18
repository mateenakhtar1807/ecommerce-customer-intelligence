"""
tests/test_project.py
-----------------------
Basic test suite covering dataset loading, preprocessing, segmentation,
purchase prediction inference, recommendations, and database operations.

Run:
    pytest tests/ -v
"""

import os
import sys
import json
import joblib
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    DATASET_PATH, PRODUCTS_PATH, DB_PATH, PURCHASE_MODEL_PATH,
    PREPROCESSING_PATH, CLUSTERING_MODEL_PATH, FEATURE_LIST_PATH
)
from src.data_preprocessing import (
    load_raw_data, clean_data, engineer_features, full_preprocessing_pipeline,
    build_feature_matrix, align_features
)
from src.segmentation import run_segmentation
from src.churn_analysis import compute_rfm
from src.recommendation import fit_tfidf, recommend_products
from src.database import init_schema, run_query


@pytest.fixture(scope="module")
def raw_df():
    assert os.path.exists(DATASET_PATH), "Run generate_dataset.py before testing"
    return load_raw_data(DATASET_PATH)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return full_preprocessing_pipeline(DATASET_PATH)


# ---------------------- Dataset loading ----------------------

def test_dataset_exists_and_loads(raw_df):
    assert len(raw_df) > 10000
    assert "purchase" in raw_df.columns


def test_dataset_has_expected_columns(raw_df):
    expected = {"customer_id", "session_id", "age", "purchase", "purchase_value",
                "product_views", "add_to_cart", "checkout_started"}
    assert expected.issubset(set(raw_df.columns))


def test_purchase_rate_realistic(raw_df):
    rate = raw_df["purchase"].mean()
    # Should not be a trivially separable / degenerate class balance
    assert 0.03 < rate < 0.5


# ---------------------- Preprocessing ----------------------

def test_clean_data_removes_missing_values(raw_df):
    cleaned = clean_data(raw_df)
    assert cleaned[["age", "session_duration", "previous_order_value", "city"]].isna().sum().sum() == 0


def test_clean_data_removes_duplicate_sessions(raw_df):
    cleaned = clean_data(raw_df)
    assert cleaned["session_id"].is_unique


def test_engineer_features_adds_columns(clean_df):
    for col in ["engagement_score", "avg_order_value", "cart_conversion_ratio",
                "is_returning_customer"]:
        assert col in clean_df.columns


def test_feature_matrix_alignment(clean_df):
    X, feature_names = build_feature_matrix(clean_df.head(50))
    # Simulate a single-row inference matrix missing some dummy categories
    single_row, _ = build_feature_matrix(clean_df.head(1))
    aligned = align_features(single_row, feature_names)
    assert list(aligned.columns) == feature_names
    assert not aligned.isna().any().any()


# ---------------------- Segmentation ----------------------

def test_segmentation_produces_expected_clusters(clean_df):
    customer_df, model, scaler, summary = run_segmentation(clean_df, n_clusters=4)
    assert customer_df["cluster"].nunique() == 4
    assert set(summary["segment_name"]) == {
        "High Value Customers", "Loyal Customers", "Potential Customers", "At-Risk Customers"
    }
    # Highest ranked segment should indeed have the highest average spend
    top_segment = summary.sort_values("value_score", ascending=False).iloc[0]
    assert top_segment["avg_spent"] == summary["avg_spent"].max()


# ---------------------- Purchase Prediction (trained model) ----------------------

@pytest.mark.skipif(not os.path.exists(PURCHASE_MODEL_PATH), reason="Run train_models.py first")
def test_purchase_model_predicts_valid_probability(clean_df):
    model = joblib.load(PURCHASE_MODEL_PATH)
    scaler = joblib.load(PREPROCESSING_PATH)
    feature_info = json.load(open(FEATURE_LIST_PATH))

    X, _ = build_feature_matrix(clean_df.head(5))
    X = align_features(X, feature_info["feature_names"])
    numeric_cols = feature_info["numeric_cols"]
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.transform(X[numeric_cols])

    proba = model.predict_proba(X_scaled)[:, 1]
    assert len(proba) == 5
    assert all(0.0 <= p <= 1.0 for p in proba)


# ---------------------- Churn / Risk ----------------------

def test_risk_scoring_buckets(clean_df):
    customer_df, _, _, _ = run_segmentation(clean_df, n_clusters=4)
    rfm = compute_rfm(customer_df)
    assert set(rfm["risk_category"].cat.categories) == {"Low Risk", "Medium Risk", "High Risk"}
    assert rfm["risk_score"].between(0, 100).all()


# ---------------------- Recommendation ----------------------

def test_recommendation_returns_top_n():
    products = pd.read_csv(PRODUCTS_PATH)
    vectorizer, tfidf_matrix, _ = fit_tfidf(products)
    recs = recommend_products(["Rings", "Bracelets"], [], products, vectorizer, tfidf_matrix, top_n=5)
    assert len(recs) == 5
    assert "similarity_score" in recs.columns
    assert recs["similarity_score"].is_monotonic_decreasing


def test_recommendation_excludes_viewed_products():
    products = pd.read_csv(PRODUCTS_PATH)
    vectorizer, tfidf_matrix, _ = fit_tfidf(products)
    viewed = [products["product_name"].iloc[0]]
    recs = recommend_products(["Rings"], viewed, products, vectorizer, tfidf_matrix, top_n=5)
    assert viewed[0] not in recs["product_name"].values


# ---------------------- Database ----------------------

def test_database_schema_and_query(tmp_path):
    test_db = str(tmp_path / "test.db")
    init_schema(test_db)
    tables = run_query(test_db, "SELECT name FROM sqlite_master WHERE type='table'")
    table_names = set(tables["name"])
    assert {"customers", "sessions", "products", "transactions"}.issubset(table_names)


@pytest.mark.skipif(not os.path.exists(DB_PATH), reason="Run setup_database.py first")
def test_populated_database_has_rows():
    customers = run_query(DB_PATH, "SELECT COUNT(*) as cnt FROM customers")
    assert customers["cnt"].iloc[0] > 0
