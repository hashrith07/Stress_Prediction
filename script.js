// =========================================================================
// REGISTER PWA SERVICE WORKER
// =========================================================================
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('./sw.js')
            .then(reg => console.log('PWA Service Worker registered:', reg.scope))
            .catch(err => console.error('PWA Service Worker registration failed:', err));
    });
}

// =========================================================================
// DOM ELEMENTS FOR OUTPUTS & CONNECTION
// =========================================================================
const placeholder = document.getElementById('result-placeholder');
const loader = document.getElementById('result-loading');
const output = document.getElementById('result-output');
const connectionBadge = document.getElementById('connection-badge');

const verdictBanner = document.getElementById('verdict-banner');
const verdictText = document.getElementById('verdict-text');

const outBpm = document.getElementById('out-bpm');
const outEda = document.getElementById('out-eda');
const outTemp = document.getElementById('out-temp');

function updateResultUI(data) {
    verdictText.textContent = data.verdict;

    if (data.stressed) {
        verdictBanner.className = 'verdict-banner stressed';
    } else {
        verdictBanner.className = 'verdict-banner relaxed';
    }

    outBpm.textContent = `${data.input_metrics.bpm} BPM`;
    outEda.textContent = `${data.input_metrics.eda.toFixed(2)} μS`;
    outTemp.textContent = `${data.input_metrics.temperature.toFixed(1)} °C`;
}

// =========================================================================
// REAL-TIME WEARABLE IOT SENSOR STREAM
// =========================================================================
const btnStream = document.getElementById('btn-stream');
const chartsArea = document.getElementById('live-charts-area');

let streamInterval = null;
let chartHr = null;
let chartEda = null;

// Initialize Chart.js Instances
function initCharts() {
    const canvasHr = document.getElementById('chart-hr');
    const canvasEda = document.getElementById('chart-eda');
    if (!canvasHr || !canvasEda) return;

    const ctxHr = canvasHr.getContext('2d');
    const ctxEda = canvasEda.getContext('2d');

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { display: false },
            y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
        },
        plugins: {
            legend: { labels: { color: '#f8fafc', font: { family: 'Outfit', size: 10 } } }
        }
    };

    chartHr = new Chart(ctxHr, {
        type: 'line',
        data: {
            labels: Array(30).fill(''),
            datasets: [{
                label: '💓 Heart Rate Waveform (BPM)',
                data: Array(30).fill(null),
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3
            }]
        },
        options: chartOptions
    });

    chartEda = new Chart(ctxEda, {
        type: 'line',
        data: {
            labels: Array(30).fill(''),
            datasets: [{
                label: '💧 Sweat Conductance / EDA (μS)',
                data: Array(30).fill(null),
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3
            }]
        },
        options: chartOptions
    });
}

function startSensorStream() {
    btnStream.textContent = "🔌 Disconnect Stream";
    btnStream.className = "btn-submit btn-stream-on";

    if (connectionBadge) {
        connectionBadge.className = "badge-status waiting";
        connectionBadge.textContent = "🟡 Connecting...";
    }

    // Show initial loading pairing state
    placeholder.classList.add('hidden');
    output.classList.add('hidden');
    loader.classList.remove('hidden');
    document.querySelector('#result-loading p').textContent = "Awaiting physical IoT Wearable transmission... (Turn on ESP32 watch)";

    if (!chartHr || !chartEda) {
        initCharts();
    }
    
    function getEffectiveApiUrl() {
        const configInput = document.getElementById('api-url');
        const configuredUrl = configInput ? configInput.value.trim() : '';
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:') {
            return 'http://localhost:8000';
        }
        return configuredUrl || 'https://stress-prediction-auws.onrender.com';
    }

    // Poll the API server every 2 seconds for live ESP32 watch updates
    const apiUrl = getEffectiveApiUrl();
    const latestUrl = `${apiUrl.replace(/\/$/, '')}/latest`;

    streamInterval = setInterval(async () => {
        try {
            const response = await fetch(latestUrl);
            if (!response.ok) throw new Error("API call failed");
            
            const result = await response.json();
            
            if (result.connected && result.data) {
                // Connection paired successfully!
                loader.classList.add('hidden');
                placeholder.classList.add('hidden');
                output.classList.remove('hidden');

                if (connectionBadge) {
                    connectionBadge.className = "badge-status connected";
                    connectionBadge.textContent = "🟢 Live IoT Stream Active";
                }

                const payload = result.data;
                
                // Update Line Graphs
                updateChart(chartHr, payload.input_metrics.bpm);
                updateChart(chartEda, parseFloat(payload.input_metrics.eda.toFixed(2)));
                
                // Update diagnostic cards Z-scores and text readouts
                updateResultUI(payload);
            } else {
                // Device is offline or went offline
                output.classList.add('hidden');
                loader.classList.remove('hidden');

                if (connectionBadge) {
                    connectionBadge.className = "badge-status waiting";
                    connectionBadge.textContent = "🟡 Awaiting Transmission";
                }

                document.querySelector('#result-loading p').textContent = result.message || "Awaiting physical IoT Wearable transmission... (Turn on ESP32 watch)";
            }
        } catch (err) {
            console.warn("Polling latest IoT stream failed:", err);
            output.classList.add('hidden');
            loader.classList.remove('hidden');

            if (connectionBadge) {
                connectionBadge.className = "badge-status waiting";
                connectionBadge.textContent = "🟡 Connecting Gateway...";
            }

            document.querySelector('#result-loading p').textContent = "Connecting to Render Cloud Gateway...";
        }
    }, 2000);
}

function updateChart(chart, newVal) {
    if (!chart) return;
    chart.data.datasets[0].data.shift();
    chart.data.datasets[0].data.push(newVal);
    chart.update('none');
}

function stopSensorStream() {
    if (streamInterval) {
        clearInterval(streamInterval);
        streamInterval = null;
    }
    btnStream.textContent = "🔌 Connect Wearable Stream";
    btnStream.className = "btn-submit btn-stream-off";
    
    if (connectionBadge) {
        connectionBadge.className = "badge-status disconnected";
        connectionBadge.textContent = "Disconnected";
    }

    // Clear charts data
    if (chartHr && chartEda) {
        chartHr.data.datasets[0].data.fill(null);
        chartEda.data.datasets[0].data.fill(null);
        chartHr.update();
        chartEda.update();
    }

    output.classList.add('hidden');
    loader.classList.add('hidden');
    placeholder.classList.remove('hidden');
}

// Toggle Stream Button Listener
btnStream.addEventListener('click', () => {
    if (streamInterval) {
        stopSensorStream();
    } else {
        startSensorStream();
    }
});

// Initialize empty chart display on load so waveforms are ready
window.addEventListener('DOMContentLoaded', () => {
    initCharts();
});

// ============================================================
// AI STRESS COMPANION - CLIENT SIDE SCRIPTING
// ============================================================
const chatBtn = document.getElementById('chat-btn');
const chatWidget = document.getElementById('chat-widget');
const closeChat = document.getElementById('close-chat');
const sendChat = document.getElementById('send-chat');
const chatInput = document.getElementById('chat-input');
const chatBody = document.getElementById('chat-body');

// Toggle chat window open/closed
chatBtn.addEventListener('click', () => {
    chatWidget.classList.toggle('active');
    if (chatWidget.classList.contains('active')) {
        chatInput.focus();
    }
});

closeChat.addEventListener('click', () => {
    chatWidget.classList.remove('active');
});

// Send message triggers
sendChat.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Display user message in UI
    appendMessage(text, 'user');
    chatInput.value = '';

    // Show typing placeholder
    const typingIndicator = appendMessage('Typing...', 'bot');

    // Retrieve the base API URL dynamically (supports local vs cloud)
    const apiUrl = getEffectiveApiUrl();
    const chatbotUrl = `${apiUrl.replace(/\/$/, '')}/chatbot`;

    try {
        const response = await fetch(chatbotUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        
        if (!response.ok) throw new Error("Network error");
        
        const data = await response.json();
        
        // Remove typing indicator and show real response
        typingIndicator.remove();
        appendMessage(data.response, 'bot');
    } catch (err) {
        console.error("Chatbot failed:", err);
        typingIndicator.remove();
        appendMessage("I'm having a connection issue, but remember to stay calm and take deep breaths.", 'bot');
    }
}

function appendMessage(text, sender) {
    const msg = document.createElement('div');
    msg.className = `message ${sender}`;
    msg.innerText = text;
    chatBody.appendChild(msg);
    chatBody.scrollTop = chatBody.scrollHeight;
    return msg;
}