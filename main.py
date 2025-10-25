from fastapi import FastAPI
from pydantic import BaseModel, validator
import pandas as pd
import joblib
from datetime import datetime
import os
from sklearn.tree import DecisionTreeClassifier
import csv

# ---------------- Constants ----------------
MODEL_FILE = 'hotel_model.pkl'
LOG_FILE = 'model_monitoring_log.csv'
MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# ---------------- Load or Initialize Model ----------------
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    model = None  # Will train when first retrain happens

# ---------------- FastAPI App ----------------
app = FastAPI(title='Hotel Booking Cancellation Prediction')

# ---------------- Input Schemas ----------------
class BookingData(BaseModel):
    lead_time: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    arrival_date_month: str
    adults: int = 0
    children: int = 0
    babies: int = 0

    # Validate month
    @validator('arrival_date_month')
    def valid_month(cls, v):
        if v not in MONTHS:
            raise ValueError(f"Invalid month: {v}. Must be one of {list(MONTHS.keys())}")
        return v

class OutcomeData(BaseModel):
    timestamp: str
    actual_is_canceled: int

# ---------------- Feature Engineering ----------------
def featureengineering(df):
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['arrival_month_num'] = df['arrival_date_month'].map(MONTHS)
    return df

# ---------------- Initialize Log File ----------------
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        "timestamp", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "arrival_date_month", "adults", "children", "babies", "prediction",
        "actual_is_canceled"
    ]).to_csv(LOG_FILE, index=False, quoting=csv.QUOTE_ALL)

# ---------------- Prediction Endpoint ----------------
@app.post('/predict')
def predict_cancellation(data: BookingData):
    global model
    if model is None:
        return {"error": "Model not trained yet. Please retrain first."}

    df = pd.DataFrame([data.dict()])
    df = featureengineering(df)
    features = ["lead_time", "total_nights", "arrival_month_num"]

    # Predict
    prediction = model.predict(df[features])
    pred_value = int(prediction[0])

    # Safe logging
    log_df = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "lead_time": data.lead_time,
        "stays_in_weekend_nights": data.stays_in_weekend_nights,
        "stays_in_week_nights": data.stays_in_week_nights,
        "arrival_date_month": data.arrival_date_month.replace(',', ''),
        "adults": data.adults,
        "children": data.children,
        "babies": data.babies,
        "prediction": pred_value,
        "actual_is_canceled": pd.NA  # Use proper NA
    }])
    log_df.to_csv(LOG_FILE, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL)

    # Alert Logic
    recent_logs = pd.read_csv(LOG_FILE, quotechar='"')
    last_n = 50
    recent = recent_logs.tail(last_n)
    cancel_rate = recent['prediction'].mean()
    if cancel_rate > 0.5:
        print(f"⚠️ ALERT: High cancellation rate detected! {cancel_rate*100:.1f}% in last {last_n} bookings")

    return {"is_canceled_prediction": pred_value}

# ---------------- Update Actual Outcome ----------------
@app.post('/update_outcome')
def update_outcome(data: OutcomeData):
    df = pd.read_csv(LOG_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    timestamp_to_update = pd.to_datetime(data.timestamp)
    df.loc[df['timestamp'] == timestamp_to_update, 'actual_is_canceled'] = int(data.actual_is_canceled)
    df.to_csv(LOG_FILE, index=False, quoting=csv.QUOTE_ALL)

    return {"status": "Actual outcome updated"}

# ---------------- Retrain Model ----------------
@app.post('/retrain')
def retrain_model():
    global model
    df = pd.read_csv(LOG_FILE)
    df = featureengineering(df)
    df = df.dropna(subset=['actual_is_canceled'])
    features = ["lead_time", "total_nights", "arrival_month_num"]
    target = "actual_is_canceled"

    if df.empty:
        return {"status": "No data available for retraining"}

    new_model = DecisionTreeClassifier()
    new_model.fit(df[features], df[target])
    joblib.dump(new_model, MODEL_FILE)

    model = new_model  # Update running model
    return {"status": "Model retrained successfully"}
