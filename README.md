# 🏠 House Price Prediction

A full-stack Python application that predicts house prices using a **Random Forest** machine learning model.

---

## Project Structure

```
house_price_prediction/
├── House Price Prediction Dataset.csv   ← original dataset
├── app.py                               ← Flask REST API (backend)
├── requirements.txt
├── model/
│   ├── train.py                         ← model training script
│   ├── model.pkl                        ← saved pipeline (generated)
│   └── feature_importance.png           ← importance chart (generated)
└── frontend/
    └── streamlit_app.py                 ← Streamlit UI (frontend)
```

---

## Dataset

| Column | Type | Description |
|---|---|---|
| Area | int | Living area in sq ft |
| Bedrooms | int | Number of bedrooms |
| Bathrooms | int | Number of bathrooms |
| Floors | int | Number of floors |
| YearBuilt | int | Year the house was built |
| Location | str | Downtown / Suburban / Urban / Rural |
| Condition | str | Excellent / Good / Fair / Poor |
| Garage | str | Yes / No |
| Price | int | **Target** — sale price in USD |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates model/model.pkl)
cd house_price_prediction
python model/train.py
```

---

## Running the Application

### Step 1 — Start the Flask API (Terminal 1)
```bash
cd house_price_prediction
python app.py
# API available at http://localhost:5000
```

### Step 2 — Start the Streamlit UI (Terminal 2)
```bash
cd house_price_prediction
streamlit run frontend/streamlit_app.py
# UI available at http://localhost:8501
```

---

## API Endpoints

### `GET /health`
Returns service status.

### `GET /model-info`
Returns model metadata, feature ranges, and valid categorical values.

### `POST /predict`
Predicts house price from input features.

**Request body (JSON):**
```json
{
  "Area": 2000,
  "Bedrooms": 3,
  "Bathrooms": 2,
  "Floors": 2,
  "YearBuilt": 1995,
  "Location": "Suburban",
  "Condition": "Good",
  "Garage": "Yes"
}
```

**Response:**
```json
{
  "predicted_price": 312450.00,
  "confidence_interval": { "low": 280000.00, "high": 345000.00 },
  "currency": "USD"
}
```

---

## Model Details

| Item | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Estimators | 300 trees |
| Preprocessing | StandardScaler (numeric) + OneHotEncoder (categorical) |
| Feature engineering | `HouseAge = 2025 - YearBuilt` |
| Train / Test split | 80 / 20 |
| Confidence interval | 5th–95th percentile of tree predictions |

---

## Frontend Features

- **Sidebar inputs** — sliders, selects and radio buttons for all features
- **Price card** — predicted price + 90% confidence interval
- **Input summary** — at-a-glance view of current selections
- **4-tab sensitivity analysis:**
  - Area sweep (line chart)
  - Bedrooms & Bathrooms (bar charts)
  - Year Built sweep (line chart)
  - Location comparison (bar chart)
