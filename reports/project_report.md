# E-Commerce Customer Intelligence & Purchase Prediction System Using Machine Learning

## Project Report

---

## 1. Title

**E-Commerce Customer Intelligence & Purchase Prediction System Using Machine Learning**

## 2. Abstract

Modern e-commerce businesses generate large volumes of customer behavioural
data — page views, cart activity, checkout events, purchase history — but
most small and mid-sized platforms lack the tooling to translate this data
into decisions. This project builds an end-to-end Data Science system that
ingests raw customer/session data, cleans and engineers features from it,
and applies both unsupervised learning (K-Means clustering for customer
segmentation) and supervised learning (Logistic Regression, Random Forest,
and XGBoost for purchase prediction) to generate actionable business
intelligence. The system also includes a transparent RFM-based customer
risk/churn score and a content-based product recommendation engine using
TF-IDF and cosine similarity. All outputs are surfaced through an
interactive Streamlit dashboard backed by a SQLite database. The project
demonstrates the complete Data Science lifecycle — from synthetic but
behaviourally-realistic data generation through to a deployable analytics
application — at a level appropriate for a B.Tech final-year evaluation.

## 3. Introduction

E-commerce platforms live and die by conversion: the percentage of visiting
sessions that convert into a completed purchase. Understanding *why* a
session converts (or doesn't) and *which customers* are worth investing
retention effort in requires more than dashboards showing raw counts — it
requires predictive and descriptive modelling. This project simulates that
real-world problem using a synthetic dataset built to mirror the
statistical properties of real e-commerce behavioural data (funnel
drop-off, class imbalance, loyalty effects, channel differences) and builds
a full analytics + ML application around it.

## 4. Problem Statement

Given customer/session-level behavioural data (device, traffic source,
engagement signals, cart/checkout activity, purchase history), can we:

1. Group customers into meaningful, actionable segments?
2. Predict, at the session level, whether a customer is likely to purchase?
3. Score customers by churn/attrition risk using a transparent method?
4. Recommend relevant products based on browsing behaviour?
5. Present all of the above through a usable business dashboard?

## 5. Motivation

Most college Data Science projects stop at "train a model, report accuracy."
Real business value comes from connecting the model outputs to decisions —
who to target with a discount, which customers are at risk of churning,
what to recommend next. This project was built to demonstrate that full
loop: data → model → business action, using multiple ML paradigms
(clustering, classification, recommendation, rule-based scoring) in one
coherent system rather than a single isolated notebook.

## 6. Objectives

- Build a realistic, well-documented synthetic e-commerce dataset.
- Implement a robust, reusable data cleaning and feature-engineering pipeline.
- Perform structured Exploratory Data Analysis covering funnel, device,
  channel, and category performance.
- Segment customers using K-Means with data-driven cluster naming.
- Train and fairly compare multiple purchase-prediction classifiers using
  imbalance-aware metrics.
- Build a transparent, business-interpretable risk-scoring module.
- Build a content-based product recommender.
- Serve everything through an interactive Streamlit dashboard backed by SQLite.

## 7. Scope

**In scope:** data generation, cleaning, EDA, feature engineering, K-Means
segmentation, supervised purchase prediction, RFM-based risk scoring,
content-based recommendation, SQLite persistence, Streamlit dashboard,
automated tests.

**Out of scope:** production deployment/hosting, payment processing,
real user authentication, live data ingestion pipelines, collaborative
filtering (would require real user-item interaction/ratings data), A/B
testing infrastructure.

## 8. Literature / Existing Systems

Real-world analogues that inspired this project's design:

- **RFM Analysis** — a long-established retail/CRM technique (Recency,
  Frequency, Monetary) for customer value segmentation, still widely used
  by e-commerce and marketing teams for its interpretability.
- **K-Means clustering** — introduced by MacQueen (1967) and one of the
  most widely used unsupervised algorithms for customer segmentation in
  industry, due to its simplicity and scalability (see scikit-learn
  documentation: https://scikit-learn.org/stable/modules/clustering.html#k-means).
- **Gradient-boosted trees (XGBoost)** — Chen & Guestrin, "XGBoost: A
  Scalable Tree Boosting System" (2016), widely used for tabular
  classification tasks including churn/purchase prediction
  (https://xgboost.readthedocs.io/).
- **TF-IDF + Cosine Similarity** — a classic information-retrieval
  technique adapted for content-based recommendation, as documented in
  scikit-learn's `TfidfVectorizer`
  (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

Existing commercial systems (Shopify Analytics, Google Analytics
Enhanced Ecommerce, Salesforce Einstein) provide similar capabilities but
as black-box, paid, cloud-hosted services — this project builds an
open, explainable, locally-runnable equivalent for educational purposes.

## 9. Proposed System

The proposed system is a modular Python application:
`generate_dataset.py` → `src/data_preprocessing.py` → `src/eda.py` →
`src/segmentation.py` / `src/purchase_prediction.py` /
`src/churn_analysis.py` / `src/recommendation.py` → `src/database.py` →
`app.py` (Streamlit). Each stage's output is a well-defined DataFrame or
serialized artifact, so stages can be tested, re-run, or swapped
independently (e.g. SQLite → MySQL, or the recommender's TF-IDF approach →
a future collaborative-filtering approach) without rewriting the rest of
the system.

## 10. System Architecture

```
DATASET
  ↓
DATA INGESTION → DATA CLEANING → EDA → FEATURE ENGINEERING
  ↓
  ├── Customer Segmentation (K-Means)
  ├── Purchase Prediction (Logistic Regression / Random Forest / XGBoost)
  ├── Churn / Risk Scoring (RFM)
  └── Product Recommendation (TF-IDF + Cosine Similarity)
  ↓
BUSINESS INSIGHTS → STREAMLIT DASHBOARD (backed by SQLite)
```

## 11. Functional Requirements

- FR1: Generate a synthetic e-commerce dataset with realistic behavioural
  correlations and controlled noise.
- FR2: Clean missing values, duplicates, outliers, and inconsistent types.
- FR3: Provide EDA visualisations for funnel, device, channel, category, and
  loyalty analysis.
- FR4: Segment customers into named clusters using K-Means.
- FR5: Predict purchase probability for a given customer/session profile.
- FR6: Score customers into Low/Medium/High risk categories.
- FR7: Recommend top-N products given viewed categories/products.
- FR8: Persist and query data via SQLite.
- FR9: Present all outputs through an interactive dashboard with filters and
  CSV export.

## 12. Non-Functional Requirements

- **Usability:** dashboard must be navigable without code by a
  non-technical evaluator.
- **Reproducibility:** fixed random seeds throughout so results are stable
  across re-runs.
- **Portability:** relative file paths only; runs identically on Windows,
  macOS, Linux.
- **Performance:** full training pipeline completes in under a minute on a
  standard laptop (12,000-row dataset).
- **Maintainability:** modular `src/` package with one responsibility per
  file; no code duplication between the dashboard and training script.

## 13. Dataset Description

`data/ecommerce_customer_data.csv` contains ~12,000 synthetic
customer/session records generated by `generate_dataset.py`, covering
demographic fields (age, gender, city), session fields (device, traffic
source, duration, pages/product/category views), funnel fields (add to
cart, cart value, checkout started), loyalty fields (previous purchases,
previous order value, days since last purchase, total spent, customer
lifetime value), and the target fields `purchase` (0/1) and
`purchase_value`. The dataset is intentionally imbalanced (~10-17%
purchase rate) and includes injected missing values, duplicate rows, and
outliers to require genuine data cleaning work, mirroring real analytics
data. A companion `data/products.csv` provides a 28-item fictional jewelry
catalogue used by the recommendation module.

## 14. Data Preprocessing

Implemented in `src/data_preprocessing.py`:

- **Duplicates:** removed by `session_id` (simulates de-duplicating
  double-fired tracking events).
- **Missing values:** numeric columns (age, session_duration,
  previous_order_value) filled with the **median** (robust to the
  right-skewed distributions present); categorical column (city) filled
  with the **mode**.
- **Outliers:** session_duration, cart_value, purchase_value, and
  total_spent are **winsorized** (capped at the 1st/99th percentile)
  rather than deleted, preserving genuine high-engagement customer rows
  which are valuable signal for the prediction model.
- **Type correction:** ages rounded to integers, binary/count columns cast
  to `int`.
- **Business-logic guardrails:** `checkout_started` forced to 0 wherever
  `add_to_cart` is 0 (a session cannot start checkout with an empty cart).

## 15. Exploratory Data Analysis

Implemented in `src/eda.py` and surfaced on the Dashboard page: purchase
vs non-purchase distribution, conversion rate by device/traffic
source/product category, session duration vs purchase (box plot), product
views vs purchase probability (line chart), add-to-cart count vs purchase
rate, previous purchases vs future purchase rate, conversion funnel,
revenue trend over time, and customer lifetime value distribution. Key
findings: checkout-started sessions convert at a dramatically higher rate
than sessions that never reach checkout; Email and Direct traffic convert
better than Paid Ads and Social Media (warm vs cold traffic); returning
customers convert meaningfully more often than first-time visitors.

## 16. Feature Engineering

Beyond the raw fields, `src/data_preprocessing.py::engineer_features`
derives: `avg_order_value` (historical spend ÷ purchase count),
`engagement_score` (a weighted combination of views, duration, cart, and
checkout signals), `cart_conversion_ratio` (add-to-cart ÷ product views),
`days_since_last_purchase_capped` (recency capped at 400 days to bound the
"never purchased" sentinel value), and `is_returning_customer` (binary
loyalty flag). Categorical columns (gender, city, device_type,
traffic_source, product_category) are one-hot encoded consistently at both
training and inference time via `build_feature_matrix` /
`align_features`, which guarantees the inference-time feature matrix has
exactly the columns the model was trained on even if a single new
customer doesn't exhibit every category.

## 17. Customer Segmentation

`src/segmentation.py` aggregates session-level data to one row per
customer, scales eight behavioural/monetary features with
`StandardScaler`, and fits K-Means (K chosen via Elbow Method and
Silhouette Score, K=4 used in the final pipeline). Rather than hardcoding
cluster names to cluster IDs (which K-Means assigns arbitrarily each run),
`name_clusters()` computes a composite z-scored "value score" per cluster
from spend, purchase count, CLV, engagement, and (inverted) recency, then
ranks clusters by that score and assigns names — **High Value Customers**,
**Loyal Customers**, **Potential Customers**, **At-Risk Customers** — in
rank order. This makes the naming logic robust to cluster-label shuffling
between training runs.

## 18. Purchase Prediction

`src/purchase_prediction.py` trains Logistic Regression, Random Forest,
and XGBoost on a stratified 80/20 train/test split, using
`class_weight="balanced"` (LR, RF) and `scale_pos_weight` (XGBoost) to
counter the ~10-17% class imbalance. All three models are evaluated with
Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix, plus
5-fold stratified cross-validation on ROC-AUC. The best model is selected
by **ROC-AUC** (not accuracy) specifically because accuracy is misleading
under class imbalance. See Section 21 for actual results.

## 19. Recommendation System

`src/recommendation.py` builds a TF-IDF vector space over each product's
category + brand + tags + description, then recommends products by
cosine similarity between a "customer profile" pseudo-document (built
from viewed categories/products) and every catalogue item. This
content-based approach was chosen over collaborative filtering because it
requires no historical user-item ratings matrix, making it usable even
for a cold-start-heavy new store.

## 20. Risk Analysis

`src/churn_analysis.py` implements an RFM-style scoring system: each
customer is quantile-scored (1-5) on Recency, Frequency, Monetary value,
and Engagement, combined into a weighted 0-100 `risk_score`, and bucketed
into Low / Medium / High Risk. This is explicitly a **business scoring
system**, not a supervised ML classifier and not a medical/financial risk
prediction — it exists to give a transparent, explainable signal a
marketing team could act on directly (e.g. targeting High Risk customers
with a win-back discount).

## 21. Model Evaluation & Results

Actual results from a full training run (`models/metrics.json`, fixed
random seed 42):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC (mean) |
|---|---|---|---|---|---|---|
| Random Forest | 0.779 | 0.307 | 0.454 | 0.366 | **0.722** | 0.734 |
| Logistic Regression | 0.674 | 0.240 | 0.608 | 0.344 | 0.721 | 0.749 |
| XGBoost | 0.757 | 0.271 | 0.433 | 0.334 | 0.694 | 0.715 |

**Random Forest** was selected as the best model (highest test-set
ROC-AUC). Precision is intentionally modest (~0.24-0.31): because purchase
is a rare event (~14% of sessions) driven by genuinely noisy behavioural
signals, a model that never over-fits to spurious patterns will naturally
trade some precision for recall once class-imbalance correction is
applied. This is a deliberately realistic result — the dataset generator
was tuned specifically to avoid producing an unrealistically perfect
(>0.95 ROC-AUC) classifier, which would suggest overfitting or feature
leakage rather than a genuine predictive model.

## 22. Dashboard

The Streamlit dashboard (`app.py`) has five pages: **Dashboard** (KPIs,
funnel, revenue trend, device/channel/category performance), **Customer
Segmentation** (cluster table, distribution, spend comparison, filters),
**Purchase Prediction** (interactive form → probability, intent label, top
contributing factors), **Product Recommendations** (category/product
picker → top-N recommended items with similarity scores), and **Customer
Risk** (risk category counts, filterable table, CSV download).

## 23. Limitations

- The dataset is synthetic; while behaviourally realistic, it does not
  capture idiosyncrasies of any real product catalogue or customer base.
- The recommender is purely content-based; it cannot capture
  "customers who bought X also bought Y" collaborative signals without
  real transaction history.
- The risk score is rule-based (RFM), not learned from labelled churn
  outcomes (no ground-truth "churned" label exists in the dataset).
- Precision on the purchase-prediction task is moderate; in a production
  setting this would be tuned against a specific business cost function
  (e.g. cost of a missed high-intent customer vs. cost of a wasted
  discount offer).

## 24. Future Scope

- Migrate persistence layer from SQLite to MySQL/PostgreSQL for
  multi-user production use.
- Add SHAP-based explainability once environment dependency constraints
  allow it.
- Incorporate real transaction data and build a hybrid
  content-based + collaborative recommender.
- Add scheduled/automated retraining and model versioning (e.g. MLflow).
- A/B test the risk-based intervention (e.g. discount targeting) against a
  control group to measure real business impact.

## 25. Conclusion

This project demonstrates a complete, working Data Science application
spanning data generation, cleaning, EDA, feature engineering, unsupervised
learning, supervised learning, model evaluation under class imbalance,
recommendation systems, and business-facing dashboarding. It intentionally
avoids inflating model performance, favouring a dataset and modelling
approach that reflects the genuine difficulty of predicting rare events
from noisy behavioural signals — the same challenge real e-commerce data
science teams face.

## 26. References

1. Scikit-learn documentation — Clustering (K-Means):
   https://scikit-learn.org/stable/modules/clustering.html#k-means
2. Scikit-learn documentation — Model evaluation metrics:
   https://scikit-learn.org/stable/modules/model_evaluation.html
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting
   System*. https://xgboost.readthedocs.io/
4. Scikit-learn documentation — TfidfVectorizer:
   https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
5. Streamlit documentation: https://docs.streamlit.io/
6. Plotly Python documentation: https://plotly.com/python/
7. Pandas documentation: https://pandas.pydata.org/docs/
8. SQLite documentation: https://www.sqlite.org/docs.html
