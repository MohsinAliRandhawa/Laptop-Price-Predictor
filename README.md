# 💻 Laptop Price Predictor — AI-Powered Market Intelligence Tool

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Model-orange?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Datasets](https://img.shields.io/badge/Training%20Data-5%2C480%20Laptops-purple)

> An end-to-end machine learning web application that predicts real-time laptop market prices using a Random Forest model trained on **5,480 global laptop listings** across 5 international datasets — with live currency conversion via REST API.

🔗 **[Live Demo → Click Here](YOUR_STREAMLIT_URL_HERE)**

---

<!-- ============================================================ -->
<!-- ADD YOUR APP SCREENSHOT HERE                                  -->
<!-- Upload a screenshot of your website to your GitHub repo      -->
<!-- then replace the URL below with the actual image path        -->
<!-- ============================================================ -->
![App Screenshot](YOUR_SCREENSHOT_URL_HERE)

---

## 🎯 Project Overview

This project demonstrates a complete **MLOps pipeline** — from raw multi-source data ingestion and feature engineering, to model training, serialization, and deployment as an interactive web application with a premium dark-mode UI.

| Component | Details |
|---|---|
| **ML Model** | Random Forest Regressor — 300 estimators |
| **Training Data** | 5,480 laptops across 5 international datasets |
| **Feature Engineering** | Currency normalization (PKR / INR / EUR → USD) |
| **Live FX Rates** | Real-time API integration (open.er-api.com) |
| **Frontend** | Streamlit with custom CSS dark theme |
| **Serialization** | Pickle — model + label encoders bundled |
| **Version Control** | Git + GitHub |

---

## 🏗️ Architecture

```
Raw CSV Datasets (5 sources)
        │
        ▼
  training.py
  ┌─────────────────────────────────┐
  │  1. Data Ingestion & Cleaning   │
  │  2. Currency Normalization      │
  │  3. Feature Engineering (19)    │
  │  4. Model Training (RF)         │
  │  5. Model Serialization (.pkl)  │
  └─────────────────────────────────┘
        │
        ▼
  laptop_price_model.pkl
        │
        ▼
  web-app.py  (Streamlit)
  ┌─────────────────────────────────┐
  │  Real-time Inference Engine     │
  │  Live USD → PKR FX API          │
  │  Interactive Premium UI         │
  └─────────────────────────────────┘
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Estimators (Trees) | 300 |
| R² Score | 0.73 |
| Mean Absolute Error | ~$274 USD |
| Training Samples | 5,480 global listings |
| Prediction Output | Point estimate + ±10% confidence range |

---

## 🌍 Data Sources

| # | Dataset | Market / Region | Currency |
|---|---|---|---|
| 1 | Laptop Price Dataset | Global | USD |
| 2 | Laptop Prices Dataset | India | INR → USD |
| 3 | Global Laptop Selling Data | Europe | EUR → USD |
| 4 | Laptop Prices Extended | Global | USD |
| 5 | PriceOye Laptops v2 | Pakistan | PKR → USD |

---

## ✨ Key Features

- 🤖 **AI Price Prediction** — 19 engineered features fed into a tuned Random Forest model
- 💱 **Live Currency Conversion** — Real-time USD → PKR via REST API with 1-hour caching
- 🌍 **Global Market Coverage** — Trained on US, EU, India, and Pakistan pricing data
- 🎨 **Premium UI** — Custom glassmorphism dark theme with CSS mesh animations
- 📊 **Confidence Interval** — ±10% price range displayed with every prediction
- 🏷️ **Price Tier Classifier** — Automatically classifies as Budget / Mid-Range / Premium / Flagship
- ⚡ **Brand Intelligence** — Brand-specific CPU, GPU, RAM, and OS filtering (e.g., Apple Silicon)

---

## 🛠️ Tech Stack

`Python 3.10+` &nbsp;·&nbsp; `Scikit-Learn` &nbsp;·&nbsp; `Streamlit` &nbsp;·&nbsp; `Pandas` &nbsp;·&nbsp; `NumPy` &nbsp;·&nbsp; `Requests` &nbsp;·&nbsp; `Pickle` &nbsp;·&nbsp; `Git`

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-train the model from scratch
python training.py

# 4. Launch the web app
streamlit run web-app.py
```

---

## 📁 Project Structure

```
├── web-app.py                    # Streamlit app — UI + inference logic
├── training.py                   # Full data pipeline + model training
├── laptop_price_model.pkl        # Serialized model bundle (model + encoders)
├── requirements.txt              # Python dependencies
├── *.csv                         # Training datasets (5 international sources)
└── README.md                     # You are here
```

---

## 👤 Author

**YOUR NAME HERE** — DevOps Engineer | ML Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](YOUR_LINKEDIN_URL_HERE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](YOUR_GITHUB_URL_HERE)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
