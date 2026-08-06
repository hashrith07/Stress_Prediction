// =========================================================================
// TAB NAVIGATION CONTROL
// =========================================================================
const tabManual = document.getElementById('tab-manual');
const tabSensor = document.getElementById('tab-sensor');
const manualForm = document.getElementById('stress-form');
const sensorPanel = document.getElementById('sensor-stream-panel');

tabManual.addEventListener('click', () => {
    tabManual.classList.add('active');
    tabSensor.classList.remove('active');
    manualForm.classList.remove('hidden');
    sensorPanel.classList.add('hidden');
    stopSensorStream(); // Ensure stream stops when switching tabs
    
    // Reset output to placeholder
    output.classList.add('hidden');
    loader.classList.add('hidden');
    placeholder.classList.remove('hidden');
});

tabSensor.addEventListener('click', () => {
    tabSensor.classList.add('active');
    tabManual.classList.remove('active');
    sensorPanel.classList.remove('hidden');
    manualForm.classList.add('hidden');
    
    // Reset output to placeholder
    output.classList.add('hidden');
    loader.classList.add('hidden');
    placeholder.classList.remove('hidden');
});

// =========================================================================
// DOM ELEMENTS FOR OUTPUTS
// =========================================================================
const placeholder = document.getElementById('result-placeholder');
const loader = document.getElementById('result-loading');
const output = document.getElementById('result-output');

const verdictBanner = document.getElementById('verdict-banner');
const verdictText = document.getElementById('verdict-text');

const outBpm = document.getElementById('out-bpm');
const outEda = document.getElementById('out-eda');
const outTemp = document.getElementById('out-temp');

const zHr = document.getElementById('z-hr');
const zEda = document.getElementById('z-eda');
const zTemp = document.getElementById('z-temp');

const mapZScoreToPercent = (z) => Math.min(100, Math.max(0, ((z + 3) / 6) * 100));

// =========================================================================
// MODE A: MANUAL DIAGNOSTICS SUBMISSION
// =========================================================================
manualForm.addEventListener('submit', async function(e) {
    e.preventDefault();

    const bpm = parseFloat(document.getElementById('bpm').value);
    const eda = parseFloat(document.getElementById('eda').value);
    const temp = parseFloat(document.getElementById('temp').value);
    const apiUrl = document.getElementById('api-url').value.trim();
    
    // Switch to Loading State
    placeholder.classList.add('hidden');
    output.classList.add('hidden');
    loader.classList.remove('hidden');
    document.querySelector('#result-loading p').textContent = "Running XGBoost Prediction Classifier...";

    try {
        const response = await fetch(`${apiUrl.replace(/\/$/, '')}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bpm, eda, temp })
        });

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const data = await response.json();

        // Render Verdict Output
        loader.classList.add('hidden');
        output.classList.remove('hidden');

        updateResultUI(data);

    } catch (error) {
        console.error('Diagnostic error:', error);
        alert(`❌ Diagnostic Failed:\n${error.message}`);
        loader.classList.add('hidden');
        placeholder.classList.remove('hidden');
    }
});

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

    zHr.style.width = `${mapZScoreToPercent(data.normalized_metrics.hr_normalized)}%`;
    zEda.style.width = `${mapZScoreToPercent(data.normalized_metrics.eda_normalized)}%`;
    zTemp.style.width = `${mapZScoreToPercent(data.normalized_metrics.temp_normalized)}%`;
}

// =========================================================================
// MODE B: LIVE SMARTWATCH STREAM SIMULATOR (POLLS PHYSICAL DEVICE DATA)
// =========================================================================
const btnStream = document.getElementById('btn-stream');
const chartsArea = document.getElementById('live-charts-area');
const subjectPicker = document.getElementById('subject-picker');

let streamInterval = null;
let chartHr = null;
let chartEda = null;

// Initialize Chart.js Instances
function initCharts() {
    const ctxHr = document.getElementById('chart-hr').getContext('2d');
    const ctxEda = document.getElementById('chart-eda').getContext('2d');

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

    // Show initial loading pairing state
    placeholder.classList.add('hidden');
    output.classList.add('hidden');
    loader.classList.remove('hidden');
    document.querySelector('#result-loading p').textContent = "Awaiting physical IoT Wearable transmission... (Turn on ESP32 watch)";

    if (!chartHr || !chartEda) {
        initCharts();
    }
    
    // Poll the Render API server every 2 seconds for live ESP32 watch updates
    const apiUrl = document.getElementById('api-url').value.trim();
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
                chartsArea.classList.remove('hidden'); // Reveal waveforms

                const payload = result.data;
                
                // Update Line Graphs
                updateChart(chartHr, payload.input_metrics.bpm);
                updateChart(chartEda, parseFloat(payload.input_metrics.eda.toFixed(2)));
                
                // Update diagnostic cards Z-scores and text readouts
                updateResultUI(payload);
            } else {
                // Device is offline or went offline
                output.classList.add('hidden');
                chartsArea.classList.add('hidden'); // Hide waveforms
                loader.classList.remove('hidden');
                document.querySelector('#result-loading p').textContent = result.message || "Awaiting physical IoT Wearable transmission... (Turn on ESP32 watch)";
            }
        } catch (err) {
            console.warn("Polling latest IoT stream failed:", err);
            output.classList.add('hidden');
            chartsArea.classList.add('hidden');
            loader.classList.remove('hidden');
            document.querySelector('#result-loading p').textContent = "Connecting to Render Cloud Gateway...";
        }
    }, 2000);
}

function updateChart(chart, newVal) {
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
    chartsArea.classList.add('hidden');
    
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