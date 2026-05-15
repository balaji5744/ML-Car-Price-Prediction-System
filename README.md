# CarVal — Flask Frontend for ML Car Price Prediction

## Project Structure

```
car_price_app/
├── app.py                          ← Flask app + /predict API
├── train_model.py                  ← One-time model training script
├── models/                         ← Created by train_model.py
│   ├── gradient_boosting_model.pkl ← Best model (CV R²=0.90, Test R²=0.94)
│   ├── preprocessor.pkl            ← Fitted ColumnTransformer
│   ├── label_encoder_model.pkl     ← Fitted LabelEncoder for 'model' col
│   └── model_classes.json          ← Car model names for UI dropdown
└── templates/
    └── index.html                  ← Jinja2 UI
```

## Setup

```bash
pip install flask numpy pandas scikit-learn xgboost
```

## Step 1 — Train & Save the Model

Run this once with your CardDekho dataset:

```bash
python train_model.py --data path/to/cardekho_imputated.csv
```

This reproduces your exact notebook pipeline:
- LabelEncoder on 'model' column
- OneHotEncoder (drop='first') on seller_type, fuel_type, transmission_type
- StandardScaler on all numerical features
- Trains Gradient Boosting with best hyperparams and saves all artefacts

## Step 2 — Run the App

```bash
python app.py
# Open http://127.0.0.1:5000
```

## Best Model

Gradient Boosting Regressor — best after RandomizedSearchCV tuning

| Metric   | Score  |
|----------|--------|
| CV R²    | 0.8963 |
| Test R²  | 0.9429 |

params: n_estimators=300, max_depth=5, lr=0.1, subsample=1.0

## Demo Mode

If ./models/ is missing, the app runs in demo mode with a warning banner.
All UI features still work for testing.
