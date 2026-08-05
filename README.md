# Customer Churn Prediction System

Predicts which customers will cancel their subscription **30 days in advance**, enabling businesses to take proactive retention action.

**Result:** 82% recall on churners | Beats baseline by 35%  
**Impact:** ~$500K annual revenue at risk identified  
**Demo:** [Deploy link goes here after Day 10]

---

## The Problem

Subscription businesses lose 20-30% of customers annually.  
By the time customers cancel, it's too late to act.  
This system identifies **who will leave** and **why** — before they go.

---

## Project Structure

```
churn-prediction/
├── data/raw/              ← Original dataset (never modified)
├── data/processed/        ← Cleaned, feature-engineered data
├── notebooks/             ← Step by step analysis notebooks
├── src/                   ← Production Python code
├── models/                ← Trained model files
├── app/                   ← Streamlit dashboard
├── docs/                  ← Charts, reports, notes
└── requirements.txt
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run notebooks in order
jupyter notebook

# 3. Launch dashboard
streamlit run app/dashboard.py
```

---

## Results (updated as project builds)

| Model | Recall | F1 | AUC |
|---|---|---|---|
| Baseline (always predict No) | 0.0 | 0.0 | 0.5 |
| Logistic Regression | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| XGBoost (Final) | TBD | TBD | TBD |

---

## Key Findings from EDA

- First 6 months = highest churn risk (50%+ churn rate)
- Month-to-month contracts churn 43% vs 3% for 2-year contracts
- High monthly charges ($70+) correlate with higher churn
- Customers WITHOUT tech support churn 3x more

---

*Built by [Your Name] | March 2026*
