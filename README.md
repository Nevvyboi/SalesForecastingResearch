# 🛒 Sales Forecasting Research
### Feature Engineering vs Deep Learning

This is my final assessment for **CSO7013 Machine Learning** at St Mary's University. The project investigates whether traditional machine learning with hand-crafted features can match deep learning performance for retail sales forecasting.

---

## 🔬 Research Question

> **Can lightweight feature engineering with gradient boosting match LSTM deep learning performance while training significantly faster?**

I hypothesised that domain-informed features (lags, rolling statistics, calendar encodings) can capture sales patterns as effectively as an LSTM's learned representations — but with a fraction of the computational cost.

---

## 📊 Dataset

I used the **UCI Online Retail Dataset** — a real-world transactional dataset containing 541,909 purchases from a UK-based online retailer (Dec 2010 – Dec 2011).

| Attribute | Value |
|-----------|-------|
| Source | UCI Machine Learning Repository |
| DOI | 10.24432/C5BW33 |
| Licence | CC BY 4.0 |
| Transactions | 541,909 |

---

## 🧪 Methodology

I compared two approaches on the same data:

### 1️⃣ LightGBM + Feature Engineering
- **20 hand-crafted features** across 5 groups:
  - **Temporal**: day_of_week, month, is_weekend, cyclical encodings
  - **Lag**: sales from 7, 14, 21, 28 days ago
  - **Rolling**: 7-day and 14-day moving averages, standard deviation
  - **External**: holidays, promotions
  - **Encoded**: target-encoded store and product family

### 2️⃣ LSTM Neural Network
- 2-layer LSTM with 32 hidden units
- 28-day input sequences
- Learns patterns directly from raw data

### Baseline
- **Seasonal Naïve**: predicts sales = last week's sales (lag-7)

---

## 📈 Key Findings

My experiments showed that **feature engineering beats deep learning** for this task:

| Model | RMSLE ↓ | Training Time |
|-------|---------|---------------|
| Seasonal Naïve | 1.04 | <1s |
| **LightGBM** | **0.42** ⭐ | **11s** |
| LSTM | 0.48 | 55s |

- LightGBM achieved **12.5% lower error** than LSTM
- LightGBM trained **5x faster**
- Lag features contributed most to performance (ablation study confirmed)

---

## 💡 Conclusion

For structured time-series forecasting like retail sales, **carefully engineered features combined with gradient boosting can outperform deep learning** while being faster, more interpretable, and easier to deploy.

This aligns with recent research (Grinsztajn et al., 2022) showing tree-based models often beat neural networks on tabular data.

---

## 📁 Project Structure

```
├── main.py                 # Run experiment
├── downloadData.py         # Fetch UCI dataset
├── src/
│   ├── dataLoader.py       # Load & preprocess data
│   ├── featureEngineer.py  # Create 20 features
│   ├── lightgbmModel.py    # LightGBM forecaster
│   ├── lstmModel.py        # LSTM neural network
│   ├── evaluator.py        # RMSLE, MAE metrics
│   └── visualizer.py       # Generate figures
└── experiments/
    ├── figures/            # Charts for paper
    └── results/            # JSON metrics
```

---

## 📚 References

- Chen, D. (2015). Online Retail [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5BW33
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
- Grinsztajn, L. et al. (2022). Why do tree-based models still outperform deep learning on tabular data? NeurIPS.

---

## 📄 Licence

MIT License — see [LICENSE](LICENSE)

---

**Student ID:** 2517238  
**Module:** CSO7013 Machine Learning  
**Institution:** St Mary's University
