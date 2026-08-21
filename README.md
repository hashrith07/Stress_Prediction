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
5. **MPU-6050 Accelerometer** (I2C motion tracking sensor)
6. **4.7kΩ Resistor** (Required pull-up for the DS18B20 digital temperature line)

---

## 🔌 Hardware Wiring Specification
The physical connections between the ESP32 development board and the sensors are configured as follows:

| Sensor Module | Sensor Pin | ESP32 Pin Destination | Wire Connection Purpose |
| :--- | :--- | :--- | :--- |
| **All Sensors Shared** | `VCC` / `+` | **`3V3`** | Shared 3.3V Power Line |
| **All Sensors Shared** | `GND` / `-` | **`GND`** | Shared Ground Reference |
| **Pulse Sensor (PPG)** | `S` (Signal) | **`VP`** (GPIO 36 / ADC1_0) | Analog Heartbeat Wave Acquisition |
| **GSR Sweat Sensor** | `SIG` (Signal)| **`D34`** (GPIO 34 / ADC1_6) | Analog Electrodermal Activity (EDA) Acquisition |
| **DS18B20 Temperature**| `DQ` (Data) | **`D25`** (GPIO 25) | OneWire Digital Temperature Stream |
| **MPU-6050 Motion** | `SCL` (Clock) | **`D22`** (GPIO 22 / I2C SCL)| I2C Synchronous Clock Line |
| | `SDA` (Data) | **`D21`** (GPIO 21 / I2C SDA)| I2C Data Transmission Line |

> [!IMPORTANT]
> The DS18B20 temperature probe requires a **4.7kΩ pull-up resistor** bridged directly between its `DQ` (Data) line and `VCC` (Power) line on the breadboard to enable the OneWire digital communication protocol.

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
├── vercel.json              # Vercel static site routing config
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

---

## 🔧 Installation & Setup

### 1. Flashing the ESP32 Wearable Board
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
