import os
import glob
import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Set Page Config for beautiful dashboard aesthetics
st.set_page_config(
    page_title="IoT Wearable Stress Predictor",
    page_icon="🫀",
    layout="wide"
)

# Custom Sleek CSS for Dark Mode glassmorphism look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    .pulse-relaxed {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10b981;
        color: #10b981;
        border-radius: 16px;
        padding: 25px;
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        animation: pulse-g 2s infinite;
    }
    .pulse-stressed {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        color: #ef4444;
        border-radius: 16px;
        padding: 25px;
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        animation: pulse-r 2s infinite;
    }
    @keyframes pulse-g {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes pulse-r {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
</style>
""", unsafe_allow_html=True)

# Helper function to parse questionnaire timelines
def parse_quest(quest_path):
    order, starts, ends = [], [], []
    with open(quest_path, 'r') as f:
        for line in f:
            if 'ORDER' in line:
                order = [p.strip() for p in line.strip().split(';')[1:] if p.strip() != '']
            elif 'START' in line:
                starts = [float(p.strip()) for p in line.strip().split(';')[1:] if p.strip() != '']
            elif 'END' in line:
                ends = [float(p.strip()) for p in line.strip().split(';')[1:] if p.strip() != '']
    return {o: (s, e) for o, s, e in zip(order, starts, ends)}

# 20 feature columns (Completely excluded ACC)
FEATURE_COLS = [
    'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range', 'eda_slope',
    'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_slope',
    'hr_mean', 'hr_std', 'hr_min', 'hr_max',
    'ibi_mean', 'ibi_sdnn', 'ibi_rmssd', 'ibi_pnn50'
]

def main():
    st.subheader("Simulated Real-Time Sensor Stream & ML Classification")
    
    # Model & Scaler Paths
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "stress_model_xgb.pkl")
    scaler_path = os.path.join(script_dir, "scaler_global.pkl")
    wesad_dir = os.path.join(script_dir, "WESAD")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(f"❌ Model or Scaler not found in {script_dir}.")
        return
    if not os.path.exists(wesad_dir):
        st.error(f"❌ WESAD Dataset directory not found at {wesad_dir}.")
        return
        
    # Load Model & Scaler directly
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
        
    # Subject selector
    subjects = sorted([os.path.basename(d) for d in glob.glob(os.path.join(wesad_dir, "S*")) if os.path.isdir(d)])
    
    # Sidebar config
    st.sidebar.title("🛠️ IoT Config")
    selected_subject = st.sidebar.selectbox("Select Subject (Simulated Wearer)", subjects)
    speed = st.sidebar.slider("Stream Speed Multiplier", 1, 10, 4, help="Fast forward the simulation time")
    
    subj_dir = os.path.join(wesad_dir, selected_subject)
    quest_file = os.path.join(subj_dir, f"{selected_subject}_quest.csv")
    e4_dir = glob.glob(os.path.join(subj_dir, "*_E4*"))[0]
    
    st.sidebar.success("✅ Device Initialized and Connected!")
    
    # Real-time dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("### 📈 Live Telemetry Waveforms")
        eda_chart_placeholder = st.empty()
        hr_chart_placeholder = st.empty()
        
    with col2:
        st.write("### 🚨 Stress Diagnostic Node")
        state_placeholder = st.empty()
        st.write("")
        metrics_placeholder = st.empty()
        
    # Start stream button
    start_btn = st.sidebar.button("🔌 Connect Wearable Device Stream")
    
    if start_btn:
        st.sidebar.info("Streaming Live Sensor Data...")
        
        # Load the CSVs for streaming playback
        df_eda = pd.read_csv(os.path.join(e4_dir, "EDA.csv"), skiprows=2, header=None)
        df_temp = pd.read_csv(os.path.join(e4_dir, "TEMP.csv"), skiprows=2, header=None)
        df_hr = pd.read_csv(os.path.join(e4_dir, "HR.csv"), skiprows=2, header=None)
        
        eda_vals = df_eda[0].values
        temp_vals = df_temp[0].values
        hr_vals = df_hr[0].values
        
        # Timeline order mapping for demonstration ground truth
        protocol = parse_quest(quest_file)
        
        # Active simulation tracking
        window_size = 15 # 15s window
        data_buffer = []
        
        # Live playback starts 30s before the TSST (Stress Test) for quick demonstration
        tsst_start_sec = int(protocol['TSST'][0] * 60) - 30
        
        step_delay = 1.0 / speed
        
        # Stream buffers for charts
        eda_history = []
        hr_history = []
        
        for t in range(tsst_start_sec, len(hr_vals)):
            # Read current values (1 sample per second playback)
            eda_curr = eda_vals[t * 4] # EDA is 4Hz
            temp_curr = temp_vals[t * 4] # TEMP is 4Hz
            hr_curr = hr_vals[t]
            
            # Ground truth label based on time
            rel_min = t / 60.0
            ground_truth = "Relaxed/Neutral"
            for cond, (s, e) in protocol.items():
                if s <= rel_min <= e:
                    if cond == "TSST":
                        ground_truth = "Stressed (Math/Public Speaking)"
                    elif cond == "Fun":
                        ground_truth = "Amused (Funny Clips)"
                    elif cond == "Base":
                        ground_truth = "Neutral Baseline"
                    break
            
            # Append to feature buffer
            data_buffer.append({
                'eda': eda_curr,
                'temp': temp_curr,
                'hr': hr_curr
            })
            
            # Maintain sliding window size
            if len(data_buffer) > window_size:
                data_buffer.pop(0)
                
            # Perform Live Feature Extraction and Classification
            if len(data_buffer) == window_size:
                df_win = pd.DataFrame(data_buffer)
                
                # Compute raw features
                eda_mean = df_win['eda'].mean()
                eda_std = df_win['eda'].std() if len(df_win) > 1 else 0.01
                eda_min = df_win['eda'].min()
                eda_max = df_win['eda'].max()
                
                temp_mean = df_win['temp'].mean()
                temp_std = df_win['temp'].std() if len(df_win) > 1 else 0.01
                temp_min = df_win['temp'].min()
                temp_max = df_win['temp'].max()
                
                hr_mean = df_win['hr'].mean()
                hr_std = df_win['hr'].std() if len(df_win) > 1 else 1.0
                hr_min = df_win['hr'].min()
                hr_max = df_win['hr'].max()
                
                # Simulate HRV parameters matching current HR
                mean_ibi = 60.0 / hr_mean
                hrv_ratio = 0.02 if hr_mean >= 85 else 0.08
                ibi_sim = mean_ibi + np.random.normal(0, mean_ibi * hrv_ratio, 15)
                ibi_mean = np.mean(ibi_sim)
                ibi_sdnn = np.std(ibi_sim)
                diff_ibi = np.diff(ibi_sim)
                ibi_rmssd = np.sqrt(np.mean(diff_ibi ** 2))
                ibi_pnn50 = (np.sum(np.abs(diff_ibi) > 0.05) / len(diff_ibi)) * 100
                
                # Create raw features mapping
                raw_feats = {
                    'eda_mean': eda_mean,
                    'eda_std': eda_std,
                    'eda_min': eda_min,
                    'eda_max': eda_max,
                    'eda_range': eda_max - eda_min,
                    'eda_slope': 0.0,
                    
                    'temp_mean': temp_mean,
                    'temp_std': temp_std,
                    'temp_min': temp_min,
                    'temp_max': temp_max,
                    'temp_range': temp_max - temp_min,
                    'temp_slope': 0.0,
                    
                    'hr_mean': hr_mean,
                    'hr_std': hr_std,
                    'hr_min': hr_min,
                    'hr_max': hr_max,
                    
                    'ibi_mean': ibi_mean,
                    'ibi_sdnn': ibi_sdnn,
                    'ibi_rmssd': ibi_rmssd,
                    'ibi_pnn50': ibi_pnn50
                }
                
                # Package raw features in correct order
                X_raw = np.zeros((1, 20))
                for idx, col in enumerate(FEATURE_COLS):
                    X_raw[0, idx] = raw_feats[col]
                    
                # Scale raw features using the fitted global scaler
                X_live = scaler.transform(X_raw)
                
                # Predict Stress
                prediction = model.predict(X_live)[0]
                
                # Render Real-time diagnosis card
                if prediction == 1:
                    state_placeholder.markdown("""
                    <div class='pulse-stressed'>
                        🚨 DANGER: STRESSED 🚨
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    state_placeholder.markdown("""
                    <div class='pulse-relaxed'>
                        🟢 SYSTEM: RELAXED
                    </div>
                    """, unsafe_allow_html=True)
                
                # Render parameters
                metrics_placeholder.markdown(f"""
                <div class='metric-card'>
                    <h4>📊 Telemetry Readouts</h4>
                    <p style='font-size: 15px; color: #94a3b8; margin-top: 5px;'>Ground Truth context: <b>{ground_truth}</b></p>
                    <div style='display: flex; justify-content: space-around; margin-top: 15px;'>
                        <div>
                            <h5>💓 Heart Rate</h5>
                            <p style='font-size: 24px; font-weight: bold; color: #ef4444;'>{hr_curr:.0f} BPM</p>
                        </div>
                        <div>
                            <h5>💧 EDA / Sweat</h5>
                            <p style='font-size: 24px; font-weight: bold; color: #10b981;'>{eda_curr:.2f} μS</p>
                        </div>
                        <div>
                            <h5>🌡️ Temp</h5>
                            <p style='font-size: 24px; font-weight: bold; color: #3b82f6;'>{temp_curr:.1f} °C</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Maintain scrolling graph history (last 100 samples)
            eda_history.append(eda_curr)
            hr_history.append(hr_curr)
            if len(eda_history) > 100:
                eda_history.pop(0)
                hr_history.pop(0)
                
            # Plot charts
            eda_chart_placeholder.line_chart(pd.DataFrame(eda_history, columns=["💧 Sweat Level / EDA (μS)"]))
            hr_chart_placeholder.line_chart(pd.DataFrame(hr_history, columns=["💓 Heart Rate (BPM)"]))
            
            # Speed control delay
            time.sleep(step_delay)

if __name__ == "__main__":
    main()
