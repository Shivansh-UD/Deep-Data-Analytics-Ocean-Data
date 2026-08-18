# 🌊 Ocean Primary Productivity Prediction

Predicting **Integrated Primary Production (IPP)** — the daily rate at which ocean phytoplankton convert sunlight and CO₂ into organic carbon — from oceanographic measurements, using machine learning and deep learning.

<p align="center">
  <img src="plots/ipp_geographic_distribution.png" width="80%" alt="Sampling locations colored by IPP">
</p>

---

## 📖 Overview

Phytoplankton are microscopic ocean plants that produce roughly **half of the oxygen on Earth** and sit at the base of the entire marine food chain. Measuring how much carbon they fix — the **ocean's primary productivity** — is critical for understanding the global carbon cycle and climate change.

This project builds a machine-learning pipeline that predicts **IPP (mgC/m²/day)** from a set of engineered oceanographic features, using real research-cruise data spanning from the Arctic to the Southern Ocean, collected between **1958 and 1994**.

The work covers the full ML lifecycle: **data cleaning → feature engineering → exploratory analysis → model training → hyperparameter tuning → deep learning comparison.**

---

## 🎯 Key Results

| Model | R² Score | RMSE |
|-------|:--------:|:----:|
| **XGBoost (Tuned)** ⭐ | **0.6455** | **0.5344** |
| Random Forest | 0.5927 | 0.5727 |
| Simple DNN (PyTorch) | 0.5166 | 0.6240 |
| Improved DNN (PyTorch) | 0.5035 | 0.6323 |
| MLP Neural Network (sklearn) | 0.4708 | 0.6528 |
| Linear Regression (baseline) | 0.3593 | 0.7184 |

> **Tuned XGBoost was the best performer.** On this tabular dataset of ~1,244 stations, gradient boosting outperformed deep learning — a well-documented pattern for structured data with limited samples.

---

## 🔬 The Data

Eight datasets from various research cruises were combined into a single source, totaling **20,558 raw measurements** across the global ocean.

| Region | Example Cruises | Period |
|--------|-----------------|--------|
| US East Coast + Arctic | Albatros, Argus, NEWP | 1975–1994 |
| Tropical Atlantic | Eumeli | 1992 |
| North Pacific (Hawaii) | HOTS | 1988–1989 |
| Equatorial Pacific | EQPAC | 1992 |
| North Atlantic (Bermuda) | BATS, bbs | 1958–1989 |
| Southern Ocean | Antiprod | 1977+ |

Each measurement records depth-resolved chlorophyll, irradiance, temperature, and photosynthesis rates at a given location and time.

---

## 🛠️ Feature Engineering

The model uses **six features** derived from the raw data to predict IPP:

| Feature | Description | Source |
|---------|-------------|--------|
| `DayOfYear` | Sequential day of year (1–365) capturing seasonality | Engineered from Day + Month |
| `Temp` | Sea surface temperature (°C) | Direct |
| `DayLength` | Solar hours per day | Direct |
| `Irr` | Surface solar irradiance | Direct |
| `Zopt` | **Euphotic zone depth** — depth where light drops to 1% | Engineered via interpolation |
| `Chlavg` | **Average chlorophyll** within the euphotic zone | Engineered |

### Two features required custom logic:

**🔦 Zopt (Euphotic Zone Depth)** — the depth where sunlight falls to 1% of the surface value, marking the bottom of the productive zone where photosynthesis can occur. Computed by linearly interpolating between the depth readings just above and below the 1% light threshold for each station.

**🌿 Chlavg (Euphotic-Zone Chlorophyll)** — the average chlorophyll concentration from the surface down to Zopt. Chlorophyll is a proxy for phytoplankton abundance, so this captures how much life is present in the productive layer. Depths below Zopt are excluded, since phytoplankton there are light-starved and inactive.

---

## 📊 Exploratory Analysis Highlights

- **Right-skewed distributions** in IPP, Chlavg, Zopt, and Irr were corrected with `log1p` transformations, normalizing the data and reducing outlier influence.
- **Chlavg** showed the strongest correlation with IPP (**r = 0.56**) — confirmed later as the most important feature.
- **Temperature** showed almost no direct linear correlation (r = 0.02), an interesting finding given that classical ocean-productivity models (VGPM, CbPM) rely heavily on it.

<p align="center">
  <img src="plots/correlation_heatmap.png" width="60%" alt="Correlation heatmap">
</p>

---

## 🤖 Modeling Approach

1. **Preprocessing** — log-transformed skewed features, then applied `StandardScaler` (fit on train only, to prevent data leakage).
2. **Train/Test Split** — 80/20 (995 train / 249 test).
3. **Baseline models** — Linear Regression, Random Forest, sklearn MLP.
4. **XGBoost** — tuned with `RandomizedSearchCV` (150 iterations, 5-fold cross-validation).
5. **Deep Learning** — two PyTorch DNNs (a simple feedforward network and an improved version with Dropout + Batch Normalization) to benchmark neural approaches.

### 🏆 Feature Importance (XGBoost)

`log_Chlavg` dominated at **~44% importance**, aligning perfectly with the correlation analysis and with oceanographic theory — phytoplankton biomass is the primary driver of productivity.

<p align="center">
  <img src="plots/feature_importance.png" width="65%" alt="XGBoost feature importance">
</p>

---

## 🧠 Key Takeaways

- **Gradient boosting beats deep learning on small tabular data.** With only ~1,244 stations, the neural networks couldn't outperform a well-tuned XGBoost — a textbook result for structured data.
- **Chlorophyll is king.** Euphotic-zone chlorophyll was consistently the most predictive feature across every model and every analysis.
- **Temperature's weak direct signal is notable.** Classical models embed temperature through an intermediate photosynthesis-rate term (Pb_opt); here, that information appears to be captured indirectly through chlorophyll and day length.

---

## 📁 Repository Structure

```
ocean-ipp-prediction/
├── README.md
├── requirements.txt
├── notebooks/
│   └── ipp_prediction.ipynb      # Full end-to-end analysis
├── src/
│   ├── data_loader.py            # Multi-file loading + cleaning
│   ├── feature_engineering.py    # Zopt & Chlavg derivation
│   └── models.py                 # ML + DL model definitions
├── data/
│   └── raw/                      # Behrenfeld-Falkowski datasets
└── plots/                        # Generated figures
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/ocean-ipp-prediction.git
cd ocean-ipp-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/ipp_prediction.ipynb
```

**Requirements:** Python 3.9+, pandas, numpy, scikit-learn, xgboost, torch, matplotlib, seaborn

---

## 🔮 Future Work

- Incorporate additional environmental predictors (mixed-layer depth, nutrient concentrations) used by established models like VGPM and CAFE.
- Expand the dataset to improve deep learning viability.
- Benchmark against the classical VGPM model (which reported R² ≈ 0.78 on ~10,000 observations).
- Explore ensemble methods combining XGBoost and Random Forest.

---

## 📚 References & Data

- Behrenfeld, M.J. & Falkowski, P.G. (1997). *A consumer's guide to phytoplankton primary productivity models.* Limnology and Oceanography.
- Ocean Productivity data — [Oregon State University Ocean Productivity Group](http://orca.science.oregonstate.edu/)

---

## 👤 Author

**Shivansh Gupta**
M.S. Data Science, University of Delaware
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)

---

<p align="center">
  <i>Built as part of ongoing ocean-productivity research using real research-cruise data.</i>
</p>