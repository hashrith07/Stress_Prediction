#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// -----------------------------------------------------
// Wi-Fi Configuration (Must be 2.4 GHz)
// -----------------------------------------------------
const char* ssid = "Excell N1 201/202";
const char* password = "excell@n1";

// -----------------------------------------------------
// Server Endpoint Configuration
// -----------------------------------------------------
// LOCAL PC SERVER (Your PC's IP on the Excell N1 Wi-Fi network):
const char* serverName = "http://192.168.0.193:8000/predict?device=iot";

// CLOUD SERVER (Uncomment if streaming directly to Render instead):
// const char* serverName = "https://stress-prediction-auws.onrender.com/predict?device=iot";

// Global variable to keep track of the demo state (false = Relaxed, true = Stressed)
bool isStressed = false;

void setup() {
  Serial.begin(115200);
  
  // Enable the onboard BOOT button (GPIO 0) as an input with internal pull-up
  pinMode(0, INPUT_PULLUP);
  
  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi Connected successfully!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Streaming to Local Server: ");
  Serial.println(serverName);
}

void loop() {
  // Check Wi-Fi connection status
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client;
    HTTPClient http;

    float bpm, temp, eda;

    if (isStressed) {
      // -----------------------------------------------------
      // Simulate STRESSED State (High heart rate, high sweat, cold temp)
      // -----------------------------------------------------
      bpm = 108.0 + (random(-30, 31) / 10.0);   // 105 - 111 BPM
      temp = 30.5 + (random(-5, 6) / 10.0);     // 30.0 - 31.0 °C
      eda = 3.65 + (random(-20, 21) / 100.0);   // 3.45 - 3.85 uS
    } else {
      // -----------------------------------------------------
      // Simulate RELAXED State (Normal heart rate, low sweat, warm temp)
      // -----------------------------------------------------
      bpm = 74.0 + (random(-20, 21) / 10.0);    // 72 - 76 BPM
      temp = 34.2 + (random(-4, 5) / 10.0);     // 33.8 - 34.6 °C
      eda = 1.35 + (random(-15, 16) / 100.0);   // 1.20 - 1.50 uS
    }

    // Package metrics into JSON payload
    StaticJsonDocument<200> doc;
    doc["bpm"] = bpm;
    doc["eda"] = eda;
    doc["temp"] = temp;

    String requestBody;
    serializeJson(doc, requestBody);

    // Send HTTP POST request to local server
    http.begin(client, serverName);
    http.addHeader("Content-Type", "application/json");

    Serial.print("Streaming live sensor payload: ");
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

  // -----------------------------------------------------
  // Responsive Button Checking Loop (Replaces delay(2000))
  // -----------------------------------------------------
  // We check the button every 10 milliseconds for a total of 2 seconds (200 times).
  // This catches any quick press instantly!
  for (int i = 0; i < 200; i++) {
    if (digitalRead(0) == LOW) {
      isStressed = !isStressed; // Toggle state
      Serial.print("\n---> [BOOT PRESSED] DEMO STATE TOGGLED! Now simulating: ");
      Serial.println(isStressed ? "🚨 STRESSED" : "🟢 RELAXED");
      delay(400); // Debounce delay
      break;      // Immediately exit the wait to send the new state to the server!
    }
    delay(10); // Check 100 times per second
  }
}
