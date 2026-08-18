"""
src/recommendation.py
-----------------------
Content-based product recommendation system.

Approach: TF-IDF vectorises each product's combined text (category + brand +
tags + description). Cosine similarity between a "customer profile" pseudo-
document (built from the categories/products they viewed) and every catalog
product ranks the most relevant products. This is a classic, interpretable
content-based recommender - no user-item ratings matrix is needed, which
suits a cold-start-heavy e-commerce scenario like a new D2C brand.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_product_corpus(products_df: pd.DataFrame) -> pd.DataFrame:
    df = products_df.copy()
    df["corpus"] = (
        df["category"].astype(str) + " " +
        df["brand"].astype(str) + " " +
        df["tags"].astype(str) + " " +
        df["description"].astype(str)
    ).str.lower()
    return df


def fit_tfidf(products_df: pd.DataFrame):
    df = build_product_corpus(products_df)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["corpus"])
    return vectorizer, tfidf_matrix, df


def recommend_products(viewed_categories, viewed_products, products_df, vectorizer, tfidf_matrix, top_n=5):
    """
    viewed_categories: list[str] of categories the customer viewed
    viewed_products: list[str] of product_names the customer viewed (optional, can be empty)
    Returns top_n recommended products (excluding ones already viewed) with similarity scores.
    """
    df = build_product_corpus(products_df)

    profile_text = " ".join(viewed_categories) + " " + " ".join(viewed_categories)  # weight category higher
    if viewed_products:
        viewed_rows = df[df["product_name"].isin(viewed_products)]
        profile_text += " " + " ".join(viewed_rows["corpus"].tolist())

    profile_text = profile_text.lower().strip()
    if not profile_text:
        # Fallback: return top overall / most common category items
        return df.head(top_n)[["product_id", "product_name", "category", "price"]].assign(similarity_score=0.0)

    profile_vec = vectorizer.transform([profile_text])
    sims = cosine_similarity(profile_vec, tfidf_matrix).flatten()

    result = df.copy()
    result["similarity_score"] = sims

    # Exclude already-viewed products from recommendations
    if viewed_products:
        result = result[~result["product_name"].isin(viewed_products)]

    result = result.sort_values("similarity_score", ascending=False).head(top_n)
    return result[["product_id", "product_name", "category", "price", "similarity_score"]].reset_index(drop=True)
