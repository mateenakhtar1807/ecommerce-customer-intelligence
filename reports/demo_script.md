# Demo Script (5-7 minutes)

## Before you start
- Have the app already running: `streamlit run app.py`
- Have a second terminal ready in case you need to show `train_models.py`
  output or `pytest` passing.

---

### 1. Dashboard (60-90 sec)
**Click:** Sidebar → "📊 Dashboard" (default page)

**Say:** "This is the business overview. We have about 12,000 sessions from
roughly 12,000 customers, a conversion rate you can see here, and total
revenue." Point at the KPI cards, then the funnel chart.

**Say:** "The funnel shows the classic e-commerce drop-off — most sessions
never even add to cart, and checkout abandonment is significant, which is
realistic." Scroll to device/traffic-source charts.

**Say:** "Email and Direct traffic convert noticeably better than Paid Ads
and Social — that's a real pattern in e-commerce, warm traffic converts
better than cold."

---

### 2. Customer Segmentation (60-90 sec)
**Click:** Sidebar → "👥 Customer Segmentation"

**Say:** "We used K-Means clustering on spend, purchase frequency,
engagement, and recency. The segment names weren't hardcoded — they're
assigned automatically based on each cluster's actual behaviour." Point at
the segment table (High Value / Loyal / Potential / At-Risk).

**Click:** the segment filter multiselect, deselect all but "At-Risk Customers"

**Say:** "I can filter down to just the At-Risk segment — this is exactly
the group a marketing team would target for a win-back campaign."

---

### 3. Purchase Prediction (90-120 sec)
**Click:** Sidebar → "🎯 Purchase Prediction"

**Say:** "This is the interactive prediction form. Let's simulate a highly
engaged customer."

**Fill in:** Product Views = 15, Add to Cart = 2, Checkout Started = Yes,
Previous Purchases = 4, Session Duration = 12

**Click:** "Predict Purchase Probability"

**Say:** "You can see the predicted probability, the intent label, and —
importantly — the top factors driving this specific prediction, which
comes from the model's feature importances. This is what makes the
prediction explainable rather than a black box."

**Optional:** change Checkout Started to "No" and re-submit to show the
probability drop.

---

### 4. Product Recommendations (45-60 sec)
**Click:** Sidebar → "💍 Product Recommendations"

**Select:** Categories = "Rings", "Bracelets"

**Click:** "Get Recommendations"

**Say:** "This uses TF-IDF and cosine similarity over the product
catalogue — no purchase history needed, so it works even for a brand new
visitor who's only browsed two categories."

---

### 5. Customer Risk (45-60 sec)
**Click:** Sidebar → "⚠️ Customer Risk"

**Say:** "This is a separate, transparent RFM-based risk score — distinct
from the ML purchase model. It's rule-based on purpose, so a
non-technical marketing team can trust and explain it."

**Click:** the risk-category filter → select only "High Risk"

**Click:** "Download filtered customer risk data (CSV)"

**Say:** "And the whole filtered list can be exported directly for a
campaign tool."

---

### 6. Business Insight Wrap-up (30 sec)
**Say:** "So in one system we've gone from raw session data, to customer
segments, to a purchase-prediction model, to a risk score, to product
recommendations — each one feeding a concrete business decision. That's
the full loop this project set out to demonstrate."

---

## Test Checklist (if asked to show code quality / testing)

- [ ] `python generate_dataset.py` runs without error, prints purchase rate
- [ ] `python setup_database.py` runs without error, prints row counts
- [ ] `python train_models.py` runs without error, prints model comparison
- [ ] `pytest tests/ -v` → all tests pass
- [ ] `streamlit run app.py` boots, all 5 pages load without error
- [ ] Purchase Prediction form produces a probability between 0-100%
- [ ] Segmentation page shows exactly 4 named segments
- [ ] Recommendation page returns exactly N products, none already "viewed"
- [ ] Customer Risk CSV download button produces a valid CSV
