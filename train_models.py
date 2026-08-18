"""
train_models.py
-----------------
End-to-end model training pipeline:
1. Load raw data
2. Clean data
3. Feature engineering
4. Train customer segmentation (K-Means)
5. Train purchase prediction models (Logistic Regression, Random Forest, XGBoost)
6. Compare & evaluate
7. Save best model + preprocessing artifacts + metrics

Run:
    python train_models.py
"""

import json
import time
import joblib
import numpy as np
import pandas as pd

from src.utils import (
    DATASET_PATH, MODELS_DIR, PURCHASE_MODEL_PATH, PREPROCESSING_PATH,
    CLUSTERING_MODEL_PATH, CLUSTER_SCALER_PATH, METRICS_PATH, FEATURE_LIST_PATH,
    ensure_dirs
)
from src.data_preprocessing import full_preprocessing_pipeline, build_feature_matrix
from src.purchase_prediction import (
    train_test_split_data, scale_numeric,
    train_logistic_regression, train_random_forest, train_xgboost,
    evaluate_model, cross_validate_model, XGBOOST_AVAILABLE
)
from src.segmentation import run_segmentation


def main():
    t0 = time.time()
    ensure_dirs()

    print("=" * 60)
    print("STEP 1-3: Load, clean, and engineer features")
    print("=" * 60)
    df = full_preprocessing_pipeline(DATASET_PATH)
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Purchase rate: {df['purchase'].mean():.2%}")

    print("\n" + "=" * 60)
    print("STEP 4: Customer Segmentation (K-Means)")
    print("=" * 60)
    customer_df, cluster_model, cluster_scaler, cluster_summary = run_segmentation(df, n_clusters=4)
    joblib.dump(cluster_model, CLUSTERING_MODEL_PATH)
    joblib.dump(cluster_scaler, CLUSTER_SCALER_PATH)
    customer_df.to_csv(f"{MODELS_DIR}/../data/customer_segments.csv", index=False)
    print(cluster_summary[["cluster", "segment_name", "size", "avg_spent", "avg_purchases"]])
    print(f"Saved clustering model -> {CLUSTERING_MODEL_PATH}")

    print("\n" + "=" * 60)
    print("STEP 5: Purchase Prediction - Feature Matrix")
    print("=" * 60)
    X, feature_names = build_feature_matrix(df)
    y = df["purchase"]
    print(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2)
    numeric_cols = [c for c in X.columns if X[c].dtype in ("float64", "int64") and X[c].nunique() > 2]
    X_train_scaled, X_test_scaled, scaler = scale_numeric(X_train, X_test, numeric_cols)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Train purchase rate: {y_train.mean():.2%} | Test purchase rate: {y_test.mean():.2%}")

    print("\n" + "=" * 60)
    print("STEP 6: Train & Compare Models")
    print("=" * 60)

    all_metrics = []

    print("\n--- Logistic Regression ---")
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    lr_metrics, lr_cm, lr_report = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    cv_mean, cv_std = cross_validate_model(lr_model, X_train_scaled, y_train, cv=5)
    lr_metrics["cv_roc_auc_mean"] = cv_mean
    lr_metrics["cv_roc_auc_std"] = cv_std
    all_metrics.append(lr_metrics)
    print(lr_metrics)

    print("\n--- Random Forest ---")
    rf_model = train_random_forest(X_train_scaled, y_train)
    rf_metrics, rf_cm, rf_report = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    cv_mean, cv_std = cross_validate_model(rf_model, X_train_scaled, y_train, cv=5)
    rf_metrics["cv_roc_auc_mean"] = cv_mean
    rf_metrics["cv_roc_auc_std"] = cv_std
    all_metrics.append(rf_metrics)
    print(rf_metrics)

    xgb_model = None
    if XGBOOST_AVAILABLE:
        print("\n--- XGBoost ---")
        xgb_model = train_xgboost(X_train_scaled, y_train)
        xgb_metrics, xgb_cm, xgb_report = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
        cv_mean, cv_std = cross_validate_model(xgb_model, X_train_scaled, y_train, cv=5)
        xgb_metrics["cv_roc_auc_mean"] = cv_mean
        xgb_metrics["cv_roc_auc_std"] = cv_std
        all_metrics.append(xgb_metrics)
        print(xgb_metrics)
    else:
        print("\nXGBoost not available - skipping.")

    print("\n" + "=" * 60)
    print("STEP 7: Model Selection")
    print("=" * 60)
    # Select best model by ROC-AUC (robust to class imbalance, unlike accuracy)
    models_lookup = {"Logistic Regression": lr_model, "Random Forest": rf_model}
    if xgb_model is not None:
        models_lookup["XGBoost"] = xgb_model

    metrics_df = pd.DataFrame(all_metrics).sort_values("roc_auc", ascending=False)
    print(metrics_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]])

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = models_lookup[best_model_name]
    print(f"\nBest model selected: {best_model_name} (highest ROC-AUC)")

    # Save best model + scaler + feature list
    joblib.dump(best_model, PURCHASE_MODEL_PATH)
    joblib.dump(scaler, PREPROCESSING_PATH)
    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump({"feature_names": feature_names, "numeric_cols": numeric_cols,
                    "best_model": best_model_name}, f, indent=2)

    metrics_summary = {
        "best_model": best_model_name,
        "all_models": metrics_df.to_dict(orient="records"),
        "training_rows": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "purchase_rate": float(df["purchase"].mean()),
        "cluster_summary": cluster_summary[["cluster", "segment_name", "size"]].to_dict(orient="records"),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2, default=str)

    print(f"\nSaved best model -> {PURCHASE_MODEL_PATH}")
    print(f"Saved preprocessing scaler -> {PREPROCESSING_PATH}")
    print(f"Saved feature list -> {FEATURE_LIST_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")

    print(f"\nTotal training time: {time.time() - t0:.1f}s")
    print("\nAll model artifacts generated successfully.")


if __name__ == "__main__":
    main()
