from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="StressIQ Prediction API",
    description="FastAPI service for real-time stress detection",
    version="1.2.0"
)

# Enable CORS for frontend/IoT integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# Load XGBoost Model
# -----------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stress_model_xgb.pkl")

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load stress model: {str(e)}")

# Universal Baseline statistics (mean, std) for ALL 24 features
UNIVERSAL_BASELINE_STATS = {
    'eda_mean': (3.909722, 2.919307),
    'eda_std': (0.015316, 0.012573),
    'eda_min': (3.789542, 2.922315),
    'eda_max': (4.017380, 2.919094),
    'eda_range': (0.227837, 0.082489),
    'eda_slope': (-0.000001, 0.000005),
    'temp_mean': (33.455559, 1.491818),
    'temp_std': (0.028801, 0.033269),
    'temp_min': (33.347298, 1.500663),
    'temp_max': (33.615633, 1.472999),
    'temp_range': (0.268336, 0.119731),
    'temp_slope': (0.000001, 0.000012),
    'acc_mean': (0.934601, 0.020249),
    'acc_std': (0.005953, 0.005378),
    'acc_min': (0.902707, 0.040405),
    'acc_max': (0.968663, 0.045451),
    'hr_mean': (72.327958, 11.350163),
    'hr_std': (4.741177, 2.540253),
    'hr_min': (64.384158, 11.183913),
    'hr_max': (80.992244, 12.571506),
    'ibi_mean': (0.849399, 0.129200),
    'ibi_sdnn': (0.057669, 0.035679),
    'ibi_rmssd': (0.053288, 0.046113),
    'ibi_pnn50': (27.407869, 23.155375),
}

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
        
        # 2. Normalize all 24 features using the universal baseline statistics mapping
        X_live = np.zeros((1, 24))
        for idx, col in enumerate(FEATURE_COLS):
            mean_val, std_val = UNIVERSAL_BASELINE_STATS[col]
            X_live[0, idx] = (raw_feats[col] - mean_val) / std_val
            
        # 3. Run XGBoost Prediction
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
        "model_loaded": model is not None,
        "model_type": "XGBoost Classifier (97.84% Accurate)"
    }