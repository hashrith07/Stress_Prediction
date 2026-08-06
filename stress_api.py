from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import time

app = FastAPI(
    title="StressIQ Prediction API",
    description="FastAPI service for real-time stress detection with live IoT device pairing",
    version="1.6.0"
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
    'hr_mean', 'hr_std', 'hr_min', 'hr_max',
    'ibi_mean', 'ibi_sdnn', 'ibi_rmssd', 'ibi_pnn50'
]

# Global state to store the latest physical IoT device reading in memory
LATEST_IOT_READING = None

# -----------------------------------------------------
# Simplified Input Schema (Only asks for BPM, EDA, and TEMP)
# -----------------------------------------------------
class StressInput(BaseModel):
    bpm: float = Field(..., description="Current Heart Rate in Beats Per Minute (BPM)", ge=0.0, le=250.0)
    eda: float = Field(..., description="Current Electrodermal Activity / Sweat Level (μS)", ge=0.0, le=100.0)
    temp: float = Field(..., description="Current Skin Temperature (°C)", ge=0.0, le=60.0)

# -----------------------------------------------------
# Prediction Endpoint (Saves latest incoming IoT reading)
# -----------------------------------------------------
@app.post("/predict")
async def predict(data: StressInput, device: str = "manual"):
    global LATEST_IOT_READING
    try:
        # =====================================================================
        # CLINICAL OVERRIDE & DATA VALIDATION GUARDS
        # =====================================================================
        clamped_temp = min(37.5, max(28.0, data.temp))
        clamped_bpm = min(220.0, max(40.0, data.bpm))
        
        # Rule 1: Extreme resting Heart Rate (Tachycardia >= 130 BPM) is always STRESSED
        if data.bpm >= 130.0:
            result = {
                "stressed": True,
                "verdict": "STRESSED",
                "confidence": "100.0% (Clinical Override)",
                "input_metrics": {
                    "bpm": data.bpm,
                    "eda": data.eda,
                    "temperature": data.temp
                },
                "synthesized_hrv": {
                    "sdnn_ms": 5.0,
                    "rmssd_ms": 7.0,
                    "pnn50_pct": 0.0
                },
                "normalized_metrics": {
                    "hr_normalized": 5.0,
                    "eda_normalized": 0.0,
                    "temp_normalized": 0.0
                }
            }
            # Cache reading for dashboard polling if it's from the physical IoT device
            if device == "iot":
                LATEST_IOT_READING = {**result, "timestamp": time.time()}
            return result
            
        # Rule 2: Active Sweat response (EDA >= 4.0 μS) + Elevated Heart Rate (>= 90 BPM) is always STRESSED
        if data.eda >= 4.0 and data.bpm >= 90.0:
            result = {
                "stressed": True,
                "verdict": "STRESSED",
                "confidence": "99.0% (Clinical Override)",
                "input_metrics": {
                    "bpm": data.bpm,
                    "eda": data.eda,
                    "temperature": data.temp
                },
                "synthesized_hrv": {
                    "sdnn_ms": 10.0,
                    "rmssd_ms": 12.0,
                    "pnn50_pct": 0.0
                },
                "normalized_metrics": {
                    "hr_normalized": 2.0,
                    "eda_normalized": 2.0,
                    "temp_normalized": 0.0
                }
            }
            # Cache reading for dashboard polling if it's from the physical IoT device
            if device == "iot":
                LATEST_IOT_READING = {**result, "timestamp": time.time()}
            return result

        # =====================================================================
        # STANDARD XGBOOST PIPELINE (Using Clamped Inputs)
        # =====================================================================
        np.random.seed(42)
        eda_sim = data.eda + np.random.normal(0, 0.02, 60)
        temp_sim = clamped_temp + np.random.normal(0, 0.05, 60)
        
        mean_ibi = 60.0 / clamped_bpm
        hrv_ratio = 0.02 if clamped_bpm >= 85 else 0.08
        ibi_sim = mean_ibi + np.random.normal(0, mean_ibi * hrv_ratio, 15)
        hr_sim = 60.0 / ibi_sim
        
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
            
            'hr_mean': np.mean(hr_sim),
            'hr_std': np.std(hr_sim),
            'hr_min': np.min(hr_sim),
            'hr_max': np.max(hr_sim),
            
            'ibi_mean': np.mean(ibi_sim),
            'ibi_sdnn': np.std(ibi_sim),
            'ibi_rmssd': np.sqrt(np.mean(np.diff(ibi_sim) ** 2)),
            'ibi_pnn50': (np.sum(np.abs(np.diff(ibi_sim)) > 0.05) / len(np.diff(ibi_sim))) * 100
        }
        
        X_raw = np.zeros((1, 20))
        for idx, col in enumerate(FEATURE_COLS):
            X_raw[0, idx] = raw_feats[col]
            
        X_live = scaler.transform(X_raw)
        
        prediction = int(model.predict(X_live)[0])
        probabilities = model.predict_proba(X_live)[0]
        confidence = float(probabilities[prediction])
        
        result = {
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
        
        # Cache reading for dashboard polling if it's from the physical IoT device
        if device == "iot":
            LATEST_IOT_READING = {**result, "timestamp": time.time()}
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------
# Live IoT Dashboard Polling Endpoint
# -----------------------------------------------------
@app.get("/latest")
async def get_latest():
    global LATEST_IOT_READING
    if LATEST_IOT_READING is None:
        return {"connected": False, "message": "No active physical device stream detected."}
        
    # Check if the reading is fresh (received in the last 12 seconds)
    is_fresh = (time.time() - LATEST_IOT_READING["timestamp"]) < 12.0
    if is_fresh:
        return {
            "connected": True,
            "data": LATEST_IOT_READING
        }
    else:
        return {
            "connected": False,
            "message": "IoT device went offline (stream timed out)."
        }

# -----------------------------------------------------
# Health Endpoint
# -----------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None and scaler is not None,
        "model_type": "Globally Standardized 3-Sensor XGBoost Pipeline (No ACC)"
    }