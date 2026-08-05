document.getElementById('stress-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    // DOM Elements
    const bpm = parseFloat(document.getElementById('bpm').value);
    const eda = parseFloat(document.getElementById('eda').value);
    const temp = document.getElementById('temp').value ? parseFloat(document.getElementById('temp').value) : null;
    const apiUrlInput = document.getElementById('api-url').value.trim();
    
    const placeholder = document.getElementById('result-placeholder');
    const loader = document.getElementById('result-loading');
    const output = document.getElementById('result-output');
    
    const verdictBanner = document.getElementById('verdict-banner');
    const verdictText = document.getElementById('verdict-text');
    const confidenceBadge = document.getElementById('confidence-badge');
    
    // Switch to Loading State
    placeholder.classList.add('hidden');
    output.classList.add('hidden');
    loader.classList.remove('hidden');

    // Build API Endpoint URL
    const predictUrl = `${apiUrlInput.replace(/\/$/, '')}/predict`;

    try {
        const response = await fetch(predictUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ bpm, eda, temp })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            const errMsg = errData.detail?.[0]?.msg || errData.error || `HTTP error ${response.status}`;
            throw new Error(errMsg);
        }

        const data = await response.json();

        // Switch to Output State
        loader.classList.add('hidden');
        output.classList.remove('hidden');

        // Render Verdict
        verdictText.textContent = data.verdict;
        confidenceBadge.textContent = `${data.confidence} Confidence`;

        if (data.stressed) {
            verdictBanner.className = 'verdict-banner stressed';
        } else {
            verdictBanner.className = 'verdict-banner relaxed';
        }

        // Render mini stats
        document.getElementById('out-bpm').textContent = `${data.input_metrics.bpm} BPM`;
        document.getElementById('out-eda').textContent = `${data.input_metrics.eda.toFixed(2)} μS`;
        document.getElementById('out-temp').textContent = `${data.input_metrics.temperature.toFixed(1)} °C`;

        // Render synthesized HRV metrics
        document.getElementById('hrv-sdnn').textContent = `${data.synthesized_hrv.sdnn_ms} ms`;
        document.getElementById('hrv-rmssd').textContent = `${data.synthesized_hrv.rmssd_ms} ms`;
        document.getElementById('hrv-pnn50').textContent = `${data.synthesized_hrv.pnn50_pct}%`;

        // Render normalized z-score bars
        // A z-score of -3 to +3 represents the normal distribution. 
        // We map z-scores from [-3, +3] to progress bar percentages [0%, 100%]
        const mapZScoreToPercent = (z) => Math.min(100, Math.max(0, ((z + 3) / 6) * 100));

        document.getElementById('z-hr').style.width = `${mapZScoreToPercent(data.normalized_metrics.hr_normalized)}%`;
        document.getElementById('z-eda').style.width = `${mapZScoreToPercent(data.normalized_metrics.eda_normalized)}%`;
        document.getElementById('z-temp').style.width = `${mapZScoreToPercent(data.normalized_metrics.temp_normalized)}%`;

    } catch (error) {
        console.error('Diagnostic error:', error);
        alert(`❌ Diagnostic Failed:\n${error.message}`);
        
        // Restore placeholder
        loader.classList.add('hidden');
        placeholder.classList.remove('hidden');
    }
});