# Quantitative Trading with Machine Learning — Supply Chain Forecasting

This project reproduces and extends the results of the Stanford paper  
**“Quantitative Trading with Machine Learning”** by *René M. Glawion (Stanford University, 2021)*.  
The original paper demonstrates that **supply chain relationships** can be used to predict stock returns — specifically **Volkswagen AG (VW)** — using data from its supplier companies.

> **Goal:** Predict daily Volkswagen (VW) stock returns using the returns of its publicly listed suppliers and macroeconomic variables.  
> **Extension (Future Work):** Apply the same methodology to the **NIFTY-50 index** and Indian supply-chain-linked stocks.

---

## 📄 Reference Paper
**Title:** Quantitative Trading with Machine Learning  
**Author:** René M. Glawion — Department of Computer Science, Stanford University  
**Year:** 2021  
**DOI / GitHub:** [rglawion/cs229_project_report](https://github.com/rglawion/cs229_project_report)

You can find the original paper (`81953230.pdf`) in the repository under `/docs`.

---

## 🧩 Project Overview

### 🎯 Objective
To predict **Volkswagen AG's daily stock returns** by leveraging:
- Returns of **36 supply chain companies** (e.g., CON.DE, BAS.DE, SIE.DE)
- **Macroeconomic indicators** like bond yields
- Lagged and rolling window features to capture short-term dynamics

---

## ⚙️ Methodology

### 1. Data
- Source: Yahoo Finance (2005–2020)
- Companies: 30 publicly traded VW suppliers (Frankfurt Stock Exchange)
- Target: VW daily return (`vw_returns`)
- Features: Supplier returns (`*_Return`), bond yields, lagged returns

### 2. Feature Engineering
- Lag features (1–5 days)
- Rolling means & standard deviations
- Normalization via `StandardScaler`

### 3. Models
We implement the same algorithms as the paper:
| Model | Type | Description |
|--------|------|--------------|
| Elastic Net | Linear Regression | Combines L1 + L2 regularization, best generalization |
| XGBoost | Tree Ensemble | Gradient boosting with regularized objective |
| LightGBM | Tree Ensemble | Efficient boosting for high-dimensional data |

---

## 🧮 Forecast Horizons
Predictions are made for **h = 1, 5, 20 days** ahead, matching the research paper.

---

## 🧾 Results

| Model | h=1 | h=5 | h=20 |
|:------|----:|----:|----:|
| **Elastic Net** | 0.0174 | 0.0401 | 0.0751 |
| **XGBoost** | 0.0176 | 0.0410 | 0.0850 |
| **LightGBM** | 0.0177 | 0.0408 | 0.0815 |

| Model | h=1 | h=5 | h=20 |
|:------|----:|----:|----:|
| **Elastic Net (time in s)** | 4.62 | 4.88 | 4.92 |
| **XGBoost (time in s)** | 92.41 | 107.03 | 109.67 |
| **LightGBM (time in s)** | 44.81 | 47.70 | 53.93 |

> 🧠 Interpretation:  
> Elastic Net achieved the lowest RMSE across all horizons, closely followed by LightGBM, replicating the paper’s main finding that **linear models outperform complex ensembles for short-term return prediction.**

---

## 📊 Backtesting
A simple trading strategy was simulated:
- Buy 1 share if predicted return > threshold `c`
- Short 1 share if predicted return < −`c`
- Compare against a simple **buy-and-hold** benchmark

> In the original paper, Elastic Net achieved up to **3× higher profits** than buy-and-hold.  
> This implementation provides the framework to reproduce similar results.

---

## 🧱 Folder Structure
vw-supplychain-forecast/
├── data/ # raw and processed CSVs
├── src/ # all scripts and models
│ ├── utils.py
│ ├── data_prep.py
│ ├── features.py
│ ├── models.py
│ ├── rolling_evaluation.py
│ ├── backtest.py
│ └── init.py
├── outputs/ # results and tables
├── models/ # saved joblib models
├── myenv/ # virtual environment
├── run_all.sh # pipeline runner
├── requirements.txt
└── README.md


---

## 🚀 How to Run
```bash
# Create virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
bash run_all.sh


Or run step-by-step:

python3 -m src.data_prep
python3 -m src.features
python3 -m src.rolling_evaluation
python3 -m src.backtest
