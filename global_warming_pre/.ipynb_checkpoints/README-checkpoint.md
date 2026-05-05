# 🌍 ClimateAI Dashboard
### Global Warming Prediction & Analysis System (1961–2022)

A production-ready, SaaS-style ML web application for analysing and predicting global temperature trends powered by **Streamlit**, **Scikit-learn**, **XGBoost**, and **LightGBM**.

---

## 🚀 Quick Start

```bash
# 1. Clone / unzip the project
cd climate_ai_dashboard

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add your dataset
#    Place the Kaggle CSV at:  data/dataset.csv
#    If absent, a realistic synthetic dataset is generated automatically.

# 5. (Optional) Pre-train & save models
python train_and_save.py

# 6. Launch the dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
climate_ai_dashboard/
│
├── app.py                  # Main Streamlit application
├── train_and_save.py       # Standalone training script
├── requirements.txt
├── README.md
│
├── data/
│   └── dataset.csv         # Place Kaggle CSV here (auto-synthetic if absent)
│
├── models/
│   ├── best_model.pkl      # Best-performing fitted model
│   ├── scaler.pkl          # StandardScaler
│   ├── label_encoder.pkl   # LabelEncoder for Country
│   └── feature_names.pkl   # Feature list
│
└── src/
    ├── __init__.py
    ├── preprocess.py       # Data cleaning, encoding, feature engineering
    ├── train.py            # Multi-model training & evaluation
    └── predict.py          # Inference utilities
```

---

## 🧠 ML Models

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Random Forest | 200 estimators |
| XGBoost | LR=0.05, depth=6, 200 rounds |
| LightGBM | LR=0.05, num_leaves=31, 200 rounds |

The best model (lowest RMSE on 20 % hold-out) is auto-selected and saved.

---

## 📊 Dashboard Sections

| Page | Contents |
|---|---|
| 🏠 Home | KPI cards, global trend chart |
| 🌍 Global Overview | Annual trends, country ranking, CO₂ scatter, country multi-select |
| 📊 EDA | Dataset preview, statistics, distributions, rolling averages, correlation heatmap |
| 🤖 Model Training | One-click training, leaderboard, feature importance |
| 🔮 Prediction | Interactive form → temperature forecast + 20-year chart + risk badge |
| 📥 Export | Download raw dataset or forecast as CSV |

---

## 🌐 Deployment

### Streamlit Community Cloud
1. Push to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app".
3. Set **Main file** to `app.py`.

### HuggingFace Spaces
1. Create a new Space (SDK = Streamlit).
2. Push all files; add `requirements.txt`.

### Render / Railway
```bash
# Procfile
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 📦 Dataset

Download **Global Warming Trends (1961–2022)** from Kaggle:  
https://www.kaggle.com/datasets/jawadawan/global-warming-trends-1961-2022

Save the CSV to `data/dataset.csv`.  
The app auto-detects columns and normalises naming variants.

---

## ⚡ Performance Notes

- `@st.cache_data` on data loading — reloads only when file changes.
- `@st.cache_resource` can wrap model loading for multi-user deployments.
- Training is triggered on demand; pre-trained models are loaded instantly on startup.

---

## 🤝 Contributing

PRs welcome! Please open an issue first for major changes.

---

## 📄 License

MIT — free to use, modify, and distribute.
