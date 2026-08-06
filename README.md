# 🧠 StressIQ: Full-Stack IoT Smartwatch Stress Predictor

**StressIQ** is an end-to-end, real-time stress diagnostics system that utilizes machine learning and wearable biosensors to classify physiological stress. Trained on the clinical **WESAD** (Wearable Stress and Affect Detection) dataset, the system processes signals from three core sensors—Heart Rate (BPM), Electrodermal Activity/Sweat (EDA), and Skin Temperature (TEMP)—to detect stress with **97.84% accuracy** using a tuned **XGBoost Classifier**.

The system features a hardware wearable layer (ESP32) that broadcasts biometrics to a cloud-deployed machine learning API (FastAPI on Render), which pairs instantly with a visual web monitoring console (Vercel).

---

## 🚀 Live Deployments
* **💻 Web Dashboard:** [https://stress-prediction-seven.vercel.app](https://stress-prediction-seven.vercel.app/)
* **📡 Cloud API Gateway:** [https://stress-prediction-auws.onrender.com/docs](https://stress-prediction-auws.onrender.com/docs)

---

## 📦 System Architecture

```
[ ESP32 Smartwatch Node ]
  ├── GSR (Sweat Sensor)
  ├── Pulse Sensor (BPM)
  └── Temp Sensor (Skin °C)
          │
          ▼ (Wi-Fi Secure Broadcast)
[ Cloud API: FastAPI on Render ] 
  ├── Loads 20-Feature XGBoost Model
  └── Global StandardScaler Pipeline
          │
          ▼ (Real-time Pairing /latest Endpoint)
[ Frontend Console: Vercel ]
  ├── 🎛️ Manual Diagnostics Calculator
  └── 📡 Live IoT Wearable Stream (Chart.js Waves)
```

---

## 🛠️ Hardware Requirements (Smartwatch Prototype)
To build the physical wearable node, the following components are used:
1. **ESP32 NodeMCU Development Board** (Wi-Fi enabled microcontroller)
2. **GSR Sensor** (Galvanic Skin Response for sweat conductance)
3. **Pulse Sensor** (Optical PPG sensor for heart rate)
4. **DS18B20 Temperature Sensor** (Waterproof skin temperature module)
5. **USB Cable & Jumper Wires**

---

## 📂 Project Structure

```
├── esp32_smartwatch.ino     # C++ Arduino firmware for ESP32 smartwatch node
├── stress_api.py            # FastAPI cloud server script (runs on Render)
├── index.html               # Frontend dashboard HTML
├── style.css                # Premium glassmorphic styling
├── script.js                # Frontend Javascript handlers and chart drawers
├── stress_model_xgb.pkl     # Retrained 20-feature XGBoost stress model (No ACC)
├── scaler_global.pkl        # Global StandardScaler for 20-feature input mapping
├── app.py                   # Streamlit app (Manual Input diagnostics)
├── app_iot_demo.py          # Streamlit app (Wearable stream replayer)
├── vercel.json              # Vercel static site routing config
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

---

## 🔧 Installation & Setup

### 1. Run the Python Dashboards locally
If you want to run the Streamlit dashboards locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive diagnostics center
streamlit run app.py

# Run the simulated IoT raw stream player
streamlit run app_iot_demo.py
```

### 2. Flashing the ESP32 Wearable Board
1. Open **`esp32_smartwatch.ino`** in the **Arduino IDE**.
2. Configure Arduino IDE:
   * Add ESP32 support: `File -> Preferences` -> Add Espressif URL.
   * Search and install **`ArduinoJson`** in the Library Manager.
3. Edit your Wi-Fi credentials in the code:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
4. Connect the ESP32 board to your computer via USB, select your board model (e.g. `ESP32 Dev Module`), select the active COM Port, and click **Upload**.

---

## 🛡️ Clinical Override & Safety Guards
To make the system robust for real-world medical anomalies, the cloud API implements a **Physiological Safety Layer** that overrides ML predictions on extreme biometrics:
* **Tachycardia Override:** Any heart rate reading **`>= 130 BPM`** instantly triggers a **`STRESSED`** classification with **100% confidence**.
* **High Sweat + Elevated Pulse:** Sweat conductance **`>= 4.0 μS`** combined with a pulse **`>= 90 BPM`** triggers a **`STRESSED`** classification.
* **Outlier Clamping:** Impossibly high skin temperature readings (e.g. **`47°C`**) are clamped to a realistic human threshold (**`37.5°C`**) before model execution to prevent false negatives.
