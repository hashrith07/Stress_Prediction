#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h> // Make sure to install this library in Arduino IDE

// -----------------------------------------------------
// Wi-Fi Configuration
// -----------------------------------------------------
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Deployed API Endpoint URL
const char* serverName = "https://stress-prediction-auws.onrender.com/predict";

// -----------------------------------------------------
// Pin Connections Configuration
// -----------------------------------------------------
const int GSR_PIN = 34;   // Analog Pin for GSR/Sweat Sensor
const int TEMP_PIN = 35;  // Analog Pin for Skin Temperature Sensor
const int PULSE_PIN = 36; // Analog Pin for optical Pulse Sensor (or I2C for MAX30102)

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi Connected successfully!");
}

void loop() {
  // Check Wi-Fi connection status
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    // 1. Read Raw Sensors
    int rawGSR = analogRead(GSR_PIN);
    int rawTemp = analogRead(TEMP_PIN);
    int rawPulse = analogRead(PULSE_PIN);

    // 2. Convert Raw Values to Physiological Metrics
    // GSR conversion to Electrodermal Activity (microSiemens)
    float eda = (float)rawGSR * (3.3 / 4095.0) * 2.0; 
    
    // Skin Temperature conversion (calibration depends on sensor module)
    float temp = ((float)rawTemp * (3.3 / 4095.0) * 100.0); 
    
    // Calculate Heart Rate (BPM) based on pulse signal peaks
    float bpm = 75.0; // Simulated constant. In production, read pulse peak intervals (IBI)
    if (rawPulse > 2000) {
      bpm = 95.0; // Rises on pulse detection
    }

    // 3. Package metrics into JSON payload
    StaticJsonDocument<200> doc;
    doc["bpm"] = bpm;
    doc["eda"] = eda;
    doc["temp"] = temp;

    String requestBody;
    serializeJson(doc, requestBody);

    // 4. Send HTTP POST request to the cloud server
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    Serial.print("Sending sensor payload: ");
    Serial.println(requestBody);

    int httpResponseCode = http.POST(requestBody);

    // 5. Read response back from Render cloud XGBoost model
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.print("API HTTP Response Code: ");
      Serial.println(httpResponseCode);
      Serial.print("Stress Classification Verdict: ");
      Serial.println(response);
    } else {
      Serial.print("Error sending POST request: ");
      Serial.println(httpResponseCode);
    }

    // Close connection
    http.end();
  } else {
    Serial.println("Wi-Fi Disconnected. Reconnecting...");
  }

  // Poll sensors and query API every 2 seconds
  delay(2000);
}
