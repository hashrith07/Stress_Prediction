#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>      // Install via Arduino IDE Library Manager
#include <OneWire.h>          // Install via Arduino IDE Library Manager
#include <DallasTemperature.h> // Install via Arduino IDE Library Manager

// -----------------------------------------------------
// Wi-Fi Configuration
// -----------------------------------------------------
const char* ssid = "Hashrith";
const char* password = "123456789";

// Deployed API Endpoint URL
const char* serverName = "https://stress-prediction-auws.onrender.com/predict?device=iot";

// -----------------------------------------------------
// Pin Connections Configuration
// -----------------------------------------------------
const int GSR_PIN = 34;   // Analog Pin for GSR/Sweat Sensor
const int TEMP_PIN = 35;  // Digital Pin for DS18B20 Temp Sensor
const int PULSE_PIN = 36; // Analog Pin for Robodo Pulse Sensor

// Setup a oneWire instance to communicate with any OneWire devices
OneWire oneWire(TEMP_PIN);

// Pass our oneWire reference to Dallas Temperature
DallasTemperature sensors(&oneWire);

void setup() {
  Serial.begin(115200);
  
  // Initialize digital temperature sensor
  sensors.begin();

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
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
    int rawPulse = analogRead(PULSE_PIN);
    
    // Request temperature reading from DS18B20
    sensors.requestTemperatures(); 
    float temp = sensors.getTempCByIndex(0); // Read skin temperature in °C

    // 2. Convert Raw GSR to Physiological Metrics (Electrodermal Activity in microSiemens)
    float eda = (float)rawGSR * (3.3 / 4095.0) * 2.0; 
    
    // 3. Simple Heart Rate calculation based on pulse signal peaks
    float bpm = 75.0; 
    if (rawPulse > 2000) {
      bpm = 95.0; // Rises on pulse detection
    }

    // 4. Package metrics into JSON payload
    StaticJsonDocument<200> doc;
    doc["bpm"] = bpm;
    doc["eda"] = eda;
    doc["temp"] = temp;

    String requestBody;
    serializeJson(doc, requestBody);

    // 5. Send HTTP POST request to the cloud server
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    Serial.print("Sending sensor payload: ");
    Serial.println(requestBody);

    int httpResponseCode = http.POST(requestBody);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.print("Server Response Code: ");
      Serial.println(httpResponseCode);
      Serial.println("Response Payload: " + response);
    } else {
      Serial.print("Error sending POST request: ");
      Serial.println(httpResponseCode);
    }

    http.end();
  } else {
    Serial.println("Wi-Fi Disconnected. Reconnecting...");
    WiFi.begin(ssid, password);
  }

  // Stream data every 2 seconds to match dashboard polling
  delay(2000);
}
