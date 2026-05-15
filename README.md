# CarVal — Flask Frontend for ML Car Price Prediction

CarVal is a machine learning-powered web application that predicts used car prices based on features like model, year, fuel type, and transmission. The project utilizes a robust regression pipeline and is deployed for real-time inference.

**Live Demo:** [https://ml-car-price-prediction-system.onrender.com]

## Project Structure

```text
car_price_app/
├── app.py                          ← Flask app + /predict API
├── train_model.py                  ← One-time model training script
├── models/                         ← Serialized artifacts
│   ├── gradient_boosting_model.pkl ← Primary production model
│   ├── preprocessor.pkl            ← Fitted ColumnTransformer
│   ├── label_encoder_model.pkl     ← Fitted LabelEncoder for 'model' col
│   └── model_classes.json          ← Car model names for UI dropdown
└── templates/
    └── index.html                  ← Jinja2 UI
```

## Model Performance & Selection

The system evaluates several regression models to ensure the highest accuracy. After hyperparameter tuning using `RandomizedSearchCV`, the **Gradient Boosting Regressor** was selected as the best performing model.

### Tuned Model Performance (Final Results)

| Model | Test R² Score | RMSE | MAE |
| --- | --- | --- | --- |
| **Gradient Boosting Regressor** | **0.9429** | **207,412** | **96,482** |
| Random Forest Regressor | 0.9394 | 213,594 | 102,131 |
| KNN Regressor | 0.9046 | 267,973 | 110,556 |
| XGBoost Regressor | 0.9023 | 271,212 | 100,205 |

*Note: Metrics derived from the final retraining phase with optimized parameters.*

### Initial Model Comparison (Baseline)

Before tuning, the following baseline performances were recorded:

* **Random Forest:** 0.9310 R²
* **XGBoost:** 0.9191 R²
* **Decision Tree:** 0.8813 R² (Exhibited overfitting)
* **Linear Regression:** 0.6645 R²

## Setup & Deployment

### Local Installation

```bash
pip install -r requirements.txt
python app.py
```

### Deployment on Render

The application is configured for deployment on Render using `gunicorn`.

1. **Build Command:** `pip install -r requirements.txt`
2. **Start Command:** `gunicorn app:app`

## Features

* **Dynamic Dropdowns:** Car models are dynamically loaded from `model_classes.json`.
* **Preprocessing Pipeline:** Includes Label Encoding for categorical names and One-Hot Encoding for seller and fuel types.
* **Robust Error Handling:** Includes a "Demo Mode" if model artifacts are missing.
