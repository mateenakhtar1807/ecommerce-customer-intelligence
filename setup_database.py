"""
setup_database.py
-------------------
Initializes the SQLite database schema and loads the cleaned dataset +
product catalogue into it.

Run:
    python setup_database.py
"""

from src.utils import DATASET_PATH, PRODUCTS_PATH, DB_PATH, ensure_dirs
from src.data_preprocessing import full_preprocessing_pipeline
from src.database import init_schema, load_data_into_db
import pandas as pd


def main():
    ensure_dirs()
    print("Initializing database schema...")
    init_schema(DB_PATH)
    print(f"Schema created at {DB_PATH}")

    print("\nLoading and cleaning dataset...")
    df = full_preprocessing_pipeline(DATASET_PATH)
    products = pd.read_csv(PRODUCTS_PATH)

    print("Loading data into database tables...")
    counts = load_data_into_db(DB_PATH, df, products)

    print("\nDatabase populated successfully:")
    for table, count in counts.items():
        print(f"  {table}: {count:,} rows")


if __name__ == "__main__":
    main()
