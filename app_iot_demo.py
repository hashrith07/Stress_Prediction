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

# Universal Baseline stats (mean, std) for the 24 features
UNIVERSAL_BASELINE_STATS = {
    'eda_mean': (3.909722, 2.919307),
    'eda_std': (0.015316, 0.012573),
    'eda_min': (3.789542, 2.922315),
    'eda_max': (4.017380, 2.919094),
    'eda_range': (0.227837, 0.082489),
    'eda_slope': (-0.000001, 0.000005),
    'temp_mean': (33.455559, 1.491818),
    'temp_std': (0.028801, 0.033269),
    'temp_min': (33.347298, 1.500663),
    'temp_max': (33.615633, 1.472999),
    'temp_range': (0.268336, 0.119731),
    'temp_slope': (0.000001, 0.000012),
    'acc_mean': (0.934601, 0.020249),
    'acc_std': (0.005953, 0.005378),
    'acc_min': (0.902707, 0.040405),
    'acc_max': (0.968663, 0.045451),
    'hr_mean': (72.327958, 11.350163),
    'hr_std': (4.741177, 2.540253),
    'hr_min': (64.384158, 11.183913),
    'hr_max': (80.992244, 12.571506),
    'ibi_mean': (0.849399, 0.129200),
    'ibi_sdnn': (0.057669, 0.035679),
    'ibi_rmssd': (0.053288, 0.046113),
    'ibi_pnn50': (27.407869, 23.155375),
}

FEATURE_COLS = [
    'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range', 'eda_slope',
    'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_slope',
    'acc_mean', 'acc_std', 'acc_min', 'acc_max',
    'hr_mean', 'hr_std', 'hr_min', 'hr_max',
    'ibi_mean', 'ibi_sdnn', 'ibi_rmssd', 'ibi_pnn50'
]

# Cache baseline profiles to speed up loading
@st.cache_data
def get_baseline_profile(subject_dir, quest_file, e4_dir):
    protocol = parse_quest(quest_file)
    base_start, base_end = protocol['Base']
    
    # Load raw values
    df_eda = pd.read_csv(os.path.join(e4_dir, "EDA.csv"), skiprows=2, header=None)
    fs_eda = float(open(os.path.join(e4_dir, "EDA.csv")).readlines()[1].strip().split(',')[0])
    
    start_idx = int(base_start * 60 * fs_eda)
    end_idx = int(base_end * 60 * fs_eda)
    baseline_eda = df_eda.iloc[start_idx:end_idx].values.flatten()
    
    # Load temp
    df_temp = pd.read_csv(os.path.join(e4_dir, "TEMP.csv"), skiprows=2, header=None)
    fs_temp = float(open(os.path.join(e4_dir, "TEMP.csv")).readlines()[1].strip().split(',')[0])
    baseline_temp = df_temp.iloc[int(base_start * 60 * fs_temp):int(base_end * 60 * fs_temp)].values.flatten()
    baseline_temp = baseline_temp[baseline_temp < 50.0]
    
    # Load HR
    df_hr = pd.read_csv(os.path.join(e4_dir, "HR.csv"), skiprows=2, header=None)
    baseline_hr = df_hr.iloc[int(base_start * 60):int(base_end * 60)].values.flatten()
    
    return {
        'eda_mean': np.mean(baseline_eda), 'eda_std': np.std(baseline_eda) if len(baseline_eda) > 1 else 0.05,
        'temp_mean': np.mean(baseline_temp), 'temp_std': np.std(baseline_temp) if len(baseline_temp) > 1 else 0.15,
        'hr_mean': np.mean(baseline_hr), 'hr_std': np.std(baseline_hr) if len(baseline_hr) > 1 else 4.0
    }

def main():
    st.title("🫀 Live IoT Smartwatch Stress Tracker Demonstration")
    st.subheader("Simulated Real-Time Sensor Stream & ML Classification")
    
    # Model Path
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "stress_model_xgb.pkl")
    wesad_dir = os.path.join(script_dir, "WESAD")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file `stress_model_xgb.pkl` (97.84% accurate XGBoost) not found in {script_dir}.")
        return
    if not os.path.exists(wesad_dir):
        st.error(f"❌ WESAD Dataset directory not found at {wesad_dir}.")
        return
        
    # Load Model directly
    model = joblib.load(model_path)
        
    # Subject selector
    subjects = sorted([os.path.basename(d) for d in glob.glob(os.path.join(wesad_dir, "S*")) if os.path.isdir(d)])
    
    # Sidebar config
    st.sidebar.title("🛠️ IoT Config")
    selected_subject = st.sidebar.selectbox("Select Subject (Simulated Wearer)", subjects)
    speed = st.sidebar.slider("Stream Speed Multiplier", 1, 10, 4, help="Fast forward the simulation time")
    
    subj_dir = os.path.join(wesad_dir, selected_subject)
    quest_file = os.path.join(subj_dir, f"{selected_subject}_quest.csv")
    e4_dir = glob.glob(os.path.join(subj_dir, "*_E4*"))[0]
    
    # Load baseline profile
    with st.spinner("Calibrating user baseline profile..."):
        baseline_profile = get_baseline_profile(subj_dir, quest_file, e4_dir)
        
    st.sidebar.success("✅ Device Calibrated to User Profile!")
    
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
                
                # Perform Baseline Normalization using Calibrated profile
                eda_mean_n = (eda_mean - baseline_profile['eda_mean']) / (baseline_profile['eda_std'] + 1e-6)
                eda_std_n = eda_std / (baseline_profile['eda_std'] + 1e-6)
                temp_mean_n = (temp_mean - baseline_profile['temp_mean']) / (baseline_profile['temp_std'] + 1e-6)
                temp_std_n = temp_std / (baseline_profile['temp_std'] + 1e-6)
                hr_mean_n = (hr_mean - baseline_profile['hr_mean']) / (baseline_profile['hr_std'] + 1e-6)
                hr_std_n = hr_std / (baseline_profile['hr_std'] + 1e-6)
                
                # Map computed features to correct indexes of our 24 feature array
                X_live = np.zeros((1, 24))
                for idx, col in enumerate(FEATURE_COLS):
                    mean, std = UNIVERSAL_BASELINE_STATS[col]
                    if col == 'eda_mean': X_live[0, idx] = eda_mean_n
                    elif col == 'eda_std': X_live[0, idx] = eda_std_n
                    elif col == 'eda_min': X_live[0, idx] = (eda_min - baseline_profile['eda_mean']) / baseline_profile['eda_std']
                    elif col == 'eda_max': X_live[0, idx] = (eda_max - baseline_profile['eda_mean']) / baseline_profile['eda_std']
                    elif col == 'temp_mean': X_live[0, idx] = temp_mean_n
                    elif col == 'temp_std': X_live[0, idx] = temp_std_n
                    elif col == 'temp_min': X_live[0, idx] = (temp_min - baseline_profile['temp_mean']) / baseline_profile['temp_std']
                    elif col == 'temp_max': X_live[0, idx] = (temp_max - baseline_profile['temp_mean']) / baseline_profile['temp_std']
                    elif col == 'hr_mean': X_live[0, idx] = hr_mean_n
                    elif col == 'hr_std': X_live[0, idx] = hr_std_n
                    elif col == 'hr_min': X_live[0, idx] = (hr_min - baseline_profile['hr_mean']) / baseline_profile['hr_std']
                    elif col == 'hr_max': X_live[0, idx] = (hr_max - baseline_profile['hr_mean']) / baseline_profile['hr_std']
                    elif col == 'ibi_mean': X_live[0, idx] = ibi_mean
                    elif col == 'ibi_sdnn': X_live[0, idx] = ibi_sdnn
                    elif col == 'ibi_rmssd': X_live[0, idx] = ibi_rmssd
                    elif col == 'ibi_pnn50': X_live[0, idx] = ibi_pnn50
                    else:
                        X_live[0, idx] = (0.95 - mean) / std # default ACC/slopes normalized
                
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
                    
                # Update metrics
                metrics_placeholder.markdown(f"""
                <div class='metric-card'>
                    <h4>Wearer Target State:</h4>
                    <p style="font-size: 20px; font-weight: bold; color: #60a5fa;">{ground_truth}</p>
                    <hr style="border: 0.5px solid rgba(255,255,255,0.1);">
                    <div style="display: flex; justify-content: space-around;">
                        <div>
                            <h5>Heart Rate</h5>
                            <p style="font-size: 24px; color: #f43f5e; font-weight: bold;">{int(hr_curr)} BPM</p>
                        </div>
                        <div>
                            <h5>Sweat Level</h5>
                            <p style="font-size: 24px; color: #10b981; font-weight: bold;">{eda_curr:.3f} μS</p>
                        </div>
                        <div>
                            <h5>Skin Temp</h5>
                            <p style="font-size: 24px; color: #3b82f6; font-weight: bold;">{temp_curr:.2f} °C</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Render sliding line charts
            eda_history.append(eda_curr)
            hr_history.append(hr_curr)
            if len(eda_history) > 60:
                eda_history.pop(0)
                hr_history.pop(0)
                
            eda_chart_placeholder.line_chart(pd.DataFrame({'EDA (Sweat level)': eda_history}), height=170)
            hr_chart_placeholder.line_chart(pd.DataFrame({'Heart Rate (BPM)': hr_history}), height=170)
            
            time.sleep(step_delay)

if __name__ == "__main__":
    main()
