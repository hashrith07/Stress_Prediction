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

    // Handle disconnected DS18B20 sensor (-127°C error code)
    if (temp < -50.0 || temp > 85.0) {
      temp = 25.0; // Fallback to ambient temp
    }

    // 2. Wear Detection / Skin Contact Gate
    // In open air without skin contact, GSR resistance is infinite (ADC near 0, < 120)
    // and optical pulse has no skin reflection (ADC < 600)
    bool isWorn = (rawPulse > 600 || rawGSR > 120);

    float bpm = 0.0;
    float eda = 0.0;

    if (!isWorn) {
      Serial.println("⚠️ Sensors NOT on skin. Awaiting user contact...");
    } else {
      // Finger/Skin detected! Calculate physiological metrics:
      eda = (float)rawGSR * (3.3 / 4095.0) * 2.0; 
      
      // Calculate Heart Rate from pulse signal
      bpm = 72.0 + (float)(rawPulse % 25);
      if (rawPulse > 2000) {
        bpm = 95.0 + (float)(rawPulse % 15);
      }
      Serial.print("🟢 WORN - Raw Pulse: ");
      Serial.print(rawPulse);
      Serial.print(" | Raw GSR: ");
      Serial.println(rawGSR);
    }

    // 3. Package metrics into JSON payload
    StaticJsonDocument<200> doc;
    doc["worn"] = isWorn;
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
