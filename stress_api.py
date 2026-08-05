from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="StressIQ Prediction API",
    description="FastAPI service for real-time stress detection using a globally standardized XGBoost pipeline",
    version="1.3.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# Load XGBoost Model & Global StandardScaler
# -----------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stress_model_xgb.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler_global.pkl")

try:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Model/Scaler not found in: {os.path.dirname(__file__)}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load stress model/scaler: {str(e)}")

FEATURE_COLS = [
    'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range', 'eda_slope',
    'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_slope',
    'acc_mean', 'acc_std', 'acc_min', 'acc_max',
    'hr_mean', 'hr_std', 'hr_min', 'hr_max',
    'ibi_mean', 'ibi_sdnn', 'ibi_rmssd', 'ibi_pnn50'
]

# -----------------------------------------------------
# Simplified Input Schema (Only asks for BPM, EDA, and TEMP)
# -----------------------------------------------------
class StressInput(BaseModel):
    bpm: float = Field(..., description="Current Heart Rate in Beats Per Minute (BPM)", ge=0.0, le=250.0)
    eda: float = Field(..., description="Current Electrodermal Activity / Sweat Level (μS)", ge=0.0, le=100.0)
    temp: float = Field(..., description="Current Skin Temperature (°C)", ge=0.0, le=60.0)

# -----------------------------------------------------
# Prediction Endpoint
# -----------------------------------------------------
@app.post("/predict")
async def predict(data: StressInput):
    try:
        # 1. Simulate 15-second physiological window around inputs with micro-noise
        np.random.seed(42)
        eda_sim = data.eda + np.random.normal(0, 0.02, 60)
        temp_sim = data.temp + np.random.normal(0, 0.05, 60)
        acc_sim = 0.95 + np.random.normal(0, 0.01, 60)  # assume quiet resting motion
        
        # Calculate HRV intervals (IBI) matching input BPM with physiological noise
        mean_ibi = 60.0 / data.bpm
        hrv_ratio = 0.02 if data.bpm >= 85 else 0.08  # Stress reduces HRV
        ibi_sim = mean_ibi + np.random.normal(0, mean_ibi * hrv_ratio, 15)
        hr_sim = 60.0 / ibi_sim
        
        # Compute the 24 raw features from the simulated window
        raw_feats = {
            'eda_mean': np.mean(eda_sim),
            'eda_std': np.std(eda_sim),
            'eda_min': np.min(eda_sim),
            'eda_max': np.max(eda_sim),
            'eda_range': np.max(eda_sim) - np.min(eda_sim),
            'eda_slope': 0.0,
            
            'temp_mean': np.mean(temp_sim),
            'temp_std': np.std(temp_sim),
            'temp_min': np.min(temp_sim),
            'temp_max': np.max(temp_sim),
            'temp_range': np.max(temp_sim) - np.min(temp_sim),
            'temp_slope': 0.0,
            
            'acc_mean': np.mean(acc_sim),
            'acc_std': np.std(acc_sim),
            'acc_min': np.min(acc_sim),
            'acc_max': np.max(acc_sim),
            
            'hr_mean': np.mean(hr_sim),
            'hr_std': np.std(hr_sim),
            'hr_min': np.min(hr_sim),
            'hr_max': np.max(hr_sim),
            
            'ibi_mean': np.mean(ibi_sim),
            'ibi_sdnn': np.std(ibi_sim),
            'ibi_rmssd': np.sqrt(np.mean(np.diff(ibi_sim) ** 2)),
            'ibi_pnn50': (np.sum(np.abs(np.diff(ibi_sim)) > 0.05) / len(np.diff(ibi_sim))) * 100
        }
        
        # 2. Package raw features in the correct order
        X_raw = np.zeros((1, 24))
        for idx, col in enumerate(FEATURE_COLS):
            X_raw[0, idx] = raw_feats[col]
            
        # 3. Scale raw features using the fitted global scaler
        X_live = scaler.transform(X_raw)
        
        # 4. Run XGBoost Prediction
        prediction = int(model.predict(X_live)[0])
        probabilities = model.predict_proba(X_live)[0]
        confidence = float(probabilities[prediction])
        
        return {
            "stressed": prediction == 1,
            "verdict": "STRESSED" if prediction == 1 else "RELAXED",
            "confidence": f"{round(confidence * 100, 2)}%",
            "input_metrics": {
                "bpm": data.bpm,
                "eda": data.eda,
                "temperature": data.temp
            },
            "synthesized_hrv": {
                "sdnn_ms": round(raw_feats['ibi_sdnn'] * 1000, 1),
                "rmssd_ms": round(raw_feats['ibi_rmssd'] * 1000, 1),
                "pnn50_pct": round(raw_feats['ibi_pnn50'], 1)
            },
            "normalized_metrics": {
                "hr_normalized": round(X_live[0, FEATURE_COLS.index('hr_mean')], 3),
                "eda_normalized": round(X_live[0, FEATURE_COLS.index('eda_mean')], 3),
                "temp_normalized": round(X_live[0, FEATURE_COLS.index('temp_mean')], 3)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------
# Health Endpoint
# -----------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None and scaler is not None,
        "model_type": "Globally Standardized XGBoost Classifier Pipeline"
    }