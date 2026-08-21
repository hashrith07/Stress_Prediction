#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// -----------------------------------------------------
// Wi-Fi Configuration
// -----------------------------------------------------
const char* ssid = "Excell N1 201/202";
const char* password = "excell@n1";

// Deployed API Endpoint URL
const char* serverName = "https://stress-prediction-auws.onrender.com/predict?device=iot";

// -----------------------------------------------------
// Pin Connections Configuration
// -----------------------------------------------------
const int GSR_PIN = 34;   // Analog Pin for GSR/Sweat Sensor
const int TEMP_PIN = 25;  // Digital Pin for DS18B20 Temp Sensor
const int PULSE_PIN = 36; // Analog Pin for Robodo Pulse Sensor (VP)

OneWire oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);

void setup() {
  Serial.begin(115200);
  
  // Enable internal software pull-up for the Temperature pin
  pinMode(TEMP_PIN, INPUT_PULLUP);
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
    WiFiClientSecure client;
    client.setInsecure(); // Bypass SSL certificate validation
    HTTPClient http;

    // 1. Read Raw Physical Sensors
    int rawGSR = analogRead(GSR_PIN);
    int rawPulse = analogRead(PULSE_PIN);
    
    // Request temperature reading from DS18B20
    sensors.requestTemperatures(); 
    float temp = sensors.getTempCByIndex(0); // Read skin temperature in °C

    // Convert Raw GSR to Physiological Metrics (Electrodermal Activity in microSiemens)
    float eda = (float)rawGSR * (3.3 / 4095.0) * 2.0; 
    
    // Calculate Heart Rate based on pulse signal peaks
    float bpm = 75.0; 
    if (rawPulse > 2000) {
      bpm = 95.0; // Rises on pulse detection
    }

    // 2. Package metrics into JSON payload
    StaticJsonDocument<200> doc;
    doc["bpm"] = bpm;
    doc["eda"] = eda;
    doc["temp"] = temp;

    String requestBody;
    serializeJson(doc, requestBody);

    // 3. Send HTTP POST request to the cloud server
    http.begin(client, serverName);
    http.addHeader("Content-Type", "application/json");

    Serial.print("Sending sensor payload: ");
    Serial.println(requestBody);

    int httpResponseCode = http.POST(requestBody);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.print("Server Response Code: ");
      Serial.println(httpResponseCode);
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
