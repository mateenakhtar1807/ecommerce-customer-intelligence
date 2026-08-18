"""
src/utils.py
-------------
Shared helper functions used across the project (path resolution, logging
helpers, formatting). Keeping these in one place avoids duplicated logic
across the Streamlit pages and training scripts.
"""

import os

# Resolve project root relative to this file so the project works no matter
# where it's launched from (no hardcoded absolute paths).
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATABASE_DIR = os.path.join(PROJECT_ROOT, "database")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

DATASET_PATH = os.path.join(DATA_DIR, "ecommerce_customer_data.csv")
PRODUCTS_PATH = os.path.join(DATA_DIR, "products.csv")
DB_PATH = os.path.join(DATABASE_DIR, "ecommerce.db")

PURCHASE_MODEL_PATH = os.path.join(MODELS_DIR, "purchase_prediction_model.pkl")
PREPROCESSING_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
CLUSTERING_MODEL_PATH = os.path.join(MODELS_DIR, "clustering_model.pkl")
CLUSTER_SCALER_PATH = os.path.join(MODELS_DIR, "cluster_scaler.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_list.json")


def ensure_dirs():
    """Create all required project directories if they don't already exist."""
    for d in [DATA_DIR, MODELS_DIR, DATABASE_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)


def format_currency(value):
    """Format a number as Indian Rupees, e.g. 12345.6 -> '₹12,345.60'."""
    try:
        return f"\u20b9{value:,.2f}"
    except (TypeError, ValueError):
        return "\u20b90.00"


def format_percent(value):
    """Format a 0-1 float as a percentage string."""
    try:
        return f"{value * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"
