# Viva Preparation — Questions & Answers

## Python

**1. Why Python for this project?**
Rich Data Science ecosystem (Pandas, NumPy, scikit-learn, Streamlit),
readable syntax, and strong community support for ML tooling.

**2. What is a virtual environment and why did you use one?**
An isolated Python installation for the project's dependencies, so package
versions don't conflict with other projects on the same machine
(`python -m venv venv`).

**3. What's the difference between a list and a NumPy array?**
Lists are heterogeneous, general-purpose Python containers. NumPy arrays
are homogeneous, contiguous in memory, and support vectorised math — much
faster for numerical operations.

**4. What is a decorator? Where might you see one in this project?**
A function that wraps another function to extend its behaviour.
`@st.cache_data` and `@st.cache_resource` in `app.py` are decorators that
cache expensive function results.

## Pandas

**5. Why use Pandas instead of raw Python for this dataset?**
Pandas gives vectorised operations, built-in missing-value handling,
groupby aggregation, and CSV I/O — all needed extensively here (cleaning,
EDA, feature engineering).

**6. Difference between `.loc` and `.iloc`?**
`.loc` selects by label (e.g. column name, index label); `.iloc` selects
by integer position.

**7. What does `groupby()` do? Give an example from this project.**
Splits data into groups by a key, applies an aggregation, and combines
results. E.g. `df.groupby("device_type")["purchase"].mean()` gives
conversion rate per device.

**8. How did you handle missing values?**
Median imputation for skewed numeric columns (age, session_duration,
previous_order_value), mode imputation for the categorical `city` column.

**9. What is `pd.get_dummies()` used for here?**
One-hot encoding categorical columns (gender, city, device_type,
traffic_source, product_category) into binary indicator columns for the
ML models.

## NumPy

**10. Why is NumPy faster than pure Python loops?**
It uses vectorised, compiled C operations under the hood instead of
Python-level loops, avoiding per-element interpreter overhead.

**11. What is broadcasting?**
NumPy's ability to apply operations between arrays of different but
compatible shapes without explicit loops (e.g. adding a scalar to an
array).

## SQL / Database

**12. Why SQLite instead of MySQL for this project?**
SQLite requires no server setup — it's a single file — making the project
trivially runnable by anyone. The schema is written in portable SQL so it
can migrate to MySQL later with minimal changes.

**13. What tables are in your schema and how do they relate?**
`customers`, `sessions` (FK → customers), `products`, and `transactions`
(FK → customers and products) — a standard normalized e-commerce schema.

**14. What is a foreign key?**
A column that references the primary key of another table, enforcing
referential integrity between related tables (e.g. `sessions.customer_id`
→ `customers.customer_id`).

**15. Why normalize the schema into 4 tables instead of one flat table?**
Avoids data duplication (e.g. customer demographics repeated per session),
keeps updates consistent, and matches how e-commerce data is genuinely
modelled in production systems.

## Machine Learning — General

**16. What is the difference between supervised and unsupervised learning?**
Supervised learning uses labelled data (purchase prediction, where
`purchase` is the known label). Unsupervised learning finds structure in
unlabelled data (K-Means clustering for segmentation, with no "correct"
cluster labels given).

**17. What is the bias-variance tradeoff?**
Bias is error from overly simple assumptions (underfitting); variance is
error from being too sensitive to training data noise (overfitting). Good
models balance both — e.g. Random Forest's `max_depth=10` and
`min_samples_leaf=5` limit variance.

**18. Why did you scale features before training?**
Logistic Regression and K-Means are distance/gradient-based and sensitive
to feature scale; unscaled monetary features (in thousands) would dominate
over ratio features (0-1 range) without `StandardScaler`.

## Classification

**19. What is a classification problem? Give this project's example.**
Predicting a discrete category/label — here, `purchase` (0 or 1) from
session features.

**20. What is class imbalance and why does it matter here?**
When one class (purchase=1, ~10-17%) is much rarer than the other. It
matters because accuracy becomes misleading — a model predicting "no
purchase" for everyone would still score ~85% accuracy while being
useless.

**21. How did you address class imbalance?**
`class_weight="balanced"` in Logistic Regression and Random Forest, and
`scale_pos_weight` in XGBoost, so misclassifying the minority class is
penalised more heavily during training.

## Logistic Regression

**22. How does Logistic Regression work?**
It models the log-odds of the positive class as a linear combination of
features, then applies a sigmoid function to output a probability between
0 and 1.

**23. Why is Logistic Regression a good baseline here?**
It's fast, interpretable (coefficients show feature direction/magnitude),
and often competitive on tabular data with mostly linear relationships.

## Random Forest

**24. How does Random Forest work?**
An ensemble of decision trees, each trained on a bootstrap sample of the
data and a random subset of features; predictions are averaged (or
majority-voted) across trees to reduce variance/overfitting.

**25. Why was Random Forest selected as the best model in your run?**
It had the highest ROC-AUC (~0.722) among the three models compared,
balancing recall and precision better than Logistic Regression (higher
recall, lower precision) and XGBoost in this run.

## XGBoost

**26. What is XGBoost and how does it differ from Random Forest?**
XGBoost is gradient-boosted trees — trees are built sequentially, each
correcting the errors of the previous ones, rather than being built
independently in parallel like Random Forest.

**27. What does `scale_pos_weight` do in XGBoost?**
Weights the minority (positive) class more heavily in the loss function
to compensate for class imbalance, computed here as
`n_negative / n_positive`.

## K-Means / Clustering

**28. How does K-Means work?**
Randomly initialises K centroids, assigns each point to its nearest
centroid, recomputes centroids as the mean of assigned points, and repeats
until convergence.

**29. How did you choose K?**
Using the Elbow Method (inertia vs K) and Silhouette Score across K=2 to
8; K=4 was used in the final pipeline, which also naturally maps to
intuitive business segments (High Value, Loyal, Potential, At-Risk).

**30. Why scale features before K-Means?**
K-Means uses Euclidean distance; unscaled features with larger numeric
ranges (like total_spent in rupees) would dominate the distance
calculation over smaller-range features.

**31. How did you name the clusters without hardcoding?**
By computing a composite z-scored "value score" per cluster from spend,
purchase count, CLV, engagement, and inverted recency, then ranking
clusters by that score and assigning names in rank order — robust to
K-Means' arbitrary cluster-ID assignment.

## Elbow Method

**32. What is the Elbow Method?**
A heuristic for choosing K: plot inertia (within-cluster sum of squared
distances) against K, and pick the K where the rate of decrease sharply
slows (the "elbow").

## Silhouette Score

**33. What is the Silhouette Score?**
A metric from -1 to 1 measuring how similar a point is to its own cluster
versus other clusters; higher values indicate better-defined clusters.

## Feature Engineering

**34. What derived features did you create and why?**
`engagement_score` (weighted combination of behavioural signals),
`avg_order_value`, `cart_conversion_ratio` (add-to-cart ÷ product views),
and `is_returning_customer` — each captures a business concept not
directly present in a single raw column.

**35. What is one-hot encoding and why use it here?**
Converting categorical variables into binary indicator columns, since ML
models require numeric input and can't directly use ordinal-free
categories like `traffic_source`.

## Evaluation Metrics

**36. What is Precision?**
Of all sessions predicted to purchase, what fraction actually did:
`TP / (TP + FP)`.

**37. What is Recall?**
Of all sessions that actually purchased, what fraction did the model
correctly identify: `TP / (TP + FN)`.

**38. What is F1 Score?**
The harmonic mean of Precision and Recall — useful when you need a single
number balancing both, especially under class imbalance.

**39. What is ROC-AUC?**
The area under the Receiver Operating Characteristic curve (True Positive
Rate vs False Positive Rate across thresholds); measures a model's overall
ability to rank positives above negatives, independent of any single
classification threshold.

**40. Why did you select the best model by ROC-AUC instead of accuracy?**
Because `purchase` is imbalanced (~10-17% positive); accuracy rewards
predicting the majority class, while ROC-AUC evaluates ranking quality
across all thresholds regardless of class balance.

**41. What does a Confusion Matrix show?**
The counts of True Positives, True Negatives, False Positives, and False
Negatives — the raw basis from which Precision, Recall, and F1 are
computed.

## Overfitting / Underfitting

**42. What is overfitting? How did you guard against it?**
A model that fits training noise rather than the underlying pattern,
performing well on training data but poorly on new data. Guarded against
via `max_depth`/`min_samples_leaf` limits (Random Forest),
regularisation-friendly Logistic Regression, and validating with a
held-out test set plus 5-fold cross-validation.

**43. What is underfitting?**
A model too simple to capture the underlying pattern, performing poorly on
both training and test data.

## Cross-Validation

**44. What is cross-validation and why did you use it?**
Splitting training data into K folds, training on K-1 and validating on
the remaining fold, repeated K times. Used here (5-fold, stratified) to
get a more robust estimate of ROC-AUC than a single train/test split
alone.

**45. Why "stratified" cross-validation specifically?**
Stratified folds preserve the original class ratio (~10-17% purchase) in
every fold, which matters for a rare-event problem — a random split could
otherwise create folds with very few or zero positive examples.

## Recommendation Systems

**46. What recommendation approach did you use and why?**
Content-based filtering (TF-IDF + cosine similarity) — chosen because it
needs no user-item ratings history, unlike collaborative filtering, making
it usable in a cold-start scenario.

**47. What is TF-IDF?**
Term Frequency-Inverse Document Frequency: a text-vectorisation technique
weighting words by how often they appear in a document relative to how
common they are across all documents, down-weighting generic terms.

**48. What is cosine similarity?**
A measure of the angle between two vectors (here, TF-IDF vectors);
values close to 1 mean the vectors point in a very similar direction
(highly similar products).

## Streamlit

**49. What is Streamlit and why was it used?**
A Python framework for building interactive data apps with minimal
front-end code — ideal for quickly building the 5-page dashboard without
writing HTML/JS/React.

**50. What do `st.cache_data` and `st.cache_resource` do?**
They cache function outputs across reruns — `st.cache_data` for
serializable data (DataFrames), `st.cache_resource` for non-serializable
objects (trained models) — avoiding expensive recomputation every time a
user interacts with a widget.

## Business Use Case

**51. What real business decision could the risk score drive?**
Targeting "High Risk" customers with a win-back discount campaign before
they churn entirely.

**52. What real business decision could the purchase prediction model drive?**
Prioritising retargeting ad spend on sessions with high predicted purchase
probability, or triggering a real-time incentive (e.g. a small discount
popup) for medium-intent sessions.

---

## 10 Difficult Questions an Examiner Might Ask

**Q1. Your ROC-AUC is only ~0.72. Isn't that a weak model?**
For a rare-event (~10-17% positive), noisy behavioural prediction
task, ROC-AUC in the 0.70-0.75 range is realistic and defensible — genuine
real-world purchase-intent models often fall in a similar range. We
deliberately tuned the synthetic data generator to avoid an inflated
>0.95 ROC-AUC, which would more likely indicate feature leakage or an
unrealistically separable dataset than a genuinely useful model. In fact,
during development we discovered and fixed exactly this: an early version
of `customer_lifetime_value` accidentally included the current session's
purchase value, causing leakage and an artificially perfect ~0.999
ROC-AUC — we removed that leakage.

**Q2. How do you know your dataset generation logic isn't just tautological — i.e., are you just encoding the answer into the features?**
The purchase outcome is drawn stochastically (`np.random.uniform < probability`)
with substantial additive noise (σ=0.07 on the probability scale) around
each funnel-stage base rate, not deterministically assigned. This means
even sessions with identical feature values can have different purchase
outcomes, which is exactly why no model achieves anywhere near 100%
accuracy despite checkout_started being a strong signal — mirroring real
cart abandonment.

**Q3. Why is precision so much lower than recall for Logistic Regression?**
`class_weight="balanced"` pushes the decision boundary to catch more true
positives (higher recall) at the cost of more false positives (lower
precision), a deliberate imbalance-correction tradeoff. Which of
precision or recall matters more depends on business cost — e.g. if
missing a genuine buyer is costlier than a wasted discount, higher recall
is preferable.

**Q4. Couldn't you have just used accuracy — why complicate things with ROC-AUC/F1?**
Accuracy is dominated by the majority class under imbalance; a trivial
"always predict no purchase" classifier would score ~85-90% accuracy while
providing zero business value. ROC-AUC and F1 specifically account for
performance on the minority (purchase) class, which is the class that
actually matters for the business decision.

**Q5. Your customer segments were auto-named — how do you know the names are actually meaningful and not just arbitrary labels?**
The names are derived from a composite value score computed directly from
each cluster's actual average spend, purchase count, CLV, engagement, and
recency — not assigned by cluster index. We verified this with a unit
test (`test_segmentation_produces_expected_clusters`) asserting the
top-ranked segment does have the highest average spend in the data.

**Q6. Why not use deep learning for purchase prediction?**
The dataset is tabular with a modest number of engineered features
(~50) and ~12,000 rows — classical models (tree ensembles, linear models)
are well-suited to this scale and are more interpretable, faster to train,
and less prone to overfitting than a neural network would be here. Deep
learning typically shows advantages on much larger datasets or
unstructured data (images, text, sequences).

**Q7. How would this system handle a genuinely new customer with no history (cold start)?**
For segmentation and risk scoring, a new customer would default to
"never purchased" sentinel values (days_since_last_purchase=9999,
previous_purchases=0), naturally placing them in a Potential/At-Risk-like
segment until they build history. For recommendations, the content-based
approach still works from viewed categories alone — no purchase history
required, unlike collaborative filtering.

**Q8. Your risk score and purchase-prediction model both use similar inputs (spend, purchases). Aren't they redundant?**
No — they answer different questions. The risk score is a transparent,
rule-based *long-term relationship health* indicator (RFM), used for
retention/marketing decisions across a customer's whole history. The
purchase-prediction model is a *session-level, short-term* purchase-intent
classifier trained on labelled outcomes, used for real-time
personalisation. They're complementary, not duplicative.

**Q9. What would break if you deployed this on real, much larger data (millions of rows)?**
Pandas' in-memory processing and single-machine K-Means/XGBoost training
would become a bottleneck; you'd move to distributed processing
(Spark/Dask), incremental/batch training, and likely swap SQLite for a
production RDBMS or data warehouse — the modular `src/` design was
intentionally built so each of those swaps is localized to one file.

**Q10. How would you validate that this system actually improves business outcomes, not just model metrics?**
Through an A/B test: deploy the model's recommendations/risk-based
interventions to a treatment group and compare actual conversion/retention
against a control group over a fixed period — model metrics like ROC-AUC
are necessary but not sufficient evidence of real business impact.
