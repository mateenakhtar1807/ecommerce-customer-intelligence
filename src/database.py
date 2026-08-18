"""
src/database.py
-----------------
SQLite database layer. Schema is written in plain, portable SQL (no
SQLite-only syntax beyond AUTOINCREMENT) so it can be migrated to MySQL
later with minimal changes (mainly: AUTOINCREMENT -> AUTO_INCREMENT,
TEXT -> VARCHAR where lengths matter).
"""

import sqlite3
import pandas as pd
import os

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    city TEXT,
    total_spent REAL,
    previous_purchases INTEGER,
    customer_lifetime_value REAL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    customer_id TEXT,
    timestamp TEXT,
    device_type TEXT,
    traffic_source TEXT,
    session_duration REAL,
    pages_viewed INTEGER,
    product_views INTEGER,
    add_to_cart INTEGER,
    checkout_started INTEGER,
    purchase INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    price REAL,
    tags TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    product_id TEXT,
    purchase_value REAL,
    timestamp TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""


def get_connection(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def init_schema(db_path):
    """Create all tables if they don't exist."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def load_data_into_db(db_path, sessions_df: pd.DataFrame, products_df: pd.DataFrame):
    """
    Populates customers, sessions, products, and transactions tables from
    the cleaned session-level dataset and the product catalogue.
    """
    conn = get_connection(db_path)
    try:
        # customers: one row per customer_id (take latest/aggregated values)
        customers = sessions_df.groupby("customer_id").agg(
            age=("age", "first"),
            gender=("gender", "first"),
            city=("city", "first"),
            total_spent=("total_spent", "max"),
            previous_purchases=("previous_purchases", "max"),
            customer_lifetime_value=("customer_lifetime_value", "max"),
        ).reset_index()
        customers.to_sql("customers", conn, if_exists="replace", index=False)

        sessions = sessions_df[[
            "session_id", "customer_id", "timestamp", "device_type", "traffic_source",
            "session_duration", "pages_viewed", "product_views", "add_to_cart",
            "checkout_started", "purchase"
        ]].copy()
        sessions["timestamp"] = sessions["timestamp"].astype(str)
        sessions.to_sql("sessions", conn, if_exists="replace", index=False)

        products = products_df[["product_id", "product_name", "category", "price", "tags", "description"]].copy()
        products.to_sql("products", conn, if_exists="replace", index=False)

        # transactions: derived from purchase==1 sessions, matched to a random product
        # in the same category as product_category for realism.
        purchases = sessions_df[sessions_df["purchase"] == 1].copy()
        import numpy as np
        rng = np.random.default_rng(42)
        product_by_category = products_df.groupby("category")["product_id"].apply(list).to_dict()

        def pick_product(cat):
            options = product_by_category.get(cat)
            if not options:
                options = products_df["product_id"].tolist()
            return rng.choice(options)

        transactions = pd.DataFrame({
            "customer_id": purchases["customer_id"].values,
            "product_id": purchases["product_category"].map(pick_product).values,
            "purchase_value": purchases["purchase_value"].values,
            "timestamp": purchases["timestamp"].astype(str).values,
        })
        transactions.to_sql("transactions", conn, if_exists="replace", index=False)

        conn.commit()
    finally:
        conn.close()

    return {
        "customers": len(customers),
        "sessions": len(sessions),
        "products": len(products),
        "transactions": len(transactions),
    }


def run_query(db_path, query, params=None):
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
