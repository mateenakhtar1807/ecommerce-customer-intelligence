"""
src/purchase_prediction.py
----------------------------
Supervised ML pipeline to predict `purchase` (0/1) and return a purchase
probability. Compares Logistic Regression, Random Forest, and XGBoost.

Because purchase=1 is a minority class (~10%), we:
- Use stratified train/test split so both sets keep the same class ratio.
- Evaluate with Precision, Recall, F1, ROC-AUC and Confusion Matrix rather
  than relying on accuracy alone (a model predicting "no purchase" for
  everyone would score ~90% accuracy while being useless).
- Use class_weight="balanced" (Logistic Regression, Random Forest) /
  scale_pos_weight (XGBoost) to counter the imbalance during training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

TARGET_COL = "purchase"

# Columns that would leak the outcome or aren't valid predictive features
LEAKAGE_COLS = ["purchase", "purchase_value", "customer_id", "session_id", "timestamp"]


def get_feature_target(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_numeric(X_train, X_test, numeric_cols):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(model, X_test, y_test, model_name=""):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return metrics, cm, report


def train_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=250, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42):
    if not XGBOOST_AVAILABLE:
        return None
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, random_state=random_state,
        eval_metric="logloss", n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def cross_validate_model(model, X, y, cv=5, random_state=42):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean(), scores.std()


def top_feature_importances(model, feature_names, top_n=8):
    """Returns top_n (feature, importance) pairs. Works for tree models and
    logistic regression (uses absolute coefficient as importance)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return []

    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def explain_prediction(model, feature_names, top_n=5):
    """
    Simple, dependency-light explainability: returns the model's globally
    most important features. (A lightweight alternative to SHAP, used here
    to avoid SHAP's heavier dependency footprint and version sensitivity
    while still giving a genuinely useful 'why' for each prediction.)
    """
    return top_feature_importances(model, feature_names, top_n=top_n)


def purchase_intent_label(probability: float) -> str:
    if probability >= 0.7:
        return "HIGH PURCHASE INTENT"
    elif probability >= 0.4:
        return "MEDIUM PURCHASE INTENT"
    else:
        return "LOW PURCHASE INTENT"
