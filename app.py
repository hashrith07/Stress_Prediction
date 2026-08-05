import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

# Set Page Config
st.set_page_config(
    page_title="StressIQ Diagnostic Center",
    page_icon="🧠",
    layout="wide"
)

# Custom Sleek CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0c0f1d;
        color: #f1f5f9;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .main-subheader {
        text-align: center;
        color: #94a3b8;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    .card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        flex: 1;
        text-align: center;
    }
    .card h3 {
        margin: 0 0 10px 0;
        font-size: 16px;
        color: #94a3b8;
    }
    .card p {
        margin: 0;
        font-size: 32px;
        font-weight: bold;
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
        margin-bottom: 20px;
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
        margin-bottom: 20px;
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

# Helper function to parse timelines
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

def run_session_analysis(e4_dir, protocol, model):
    df_eda = pd.read_csv(os.path.join(e4_dir, "EDA.csv"), skiprows=2, header=None)
    df_temp = pd.read_csv(os.path.join(e4_dir, "TEMP.csv"), skiprows=2, header=None)
    df_hr = pd.read_csv(os.path.join(e4_dir, "HR.csv"), skiprows=2, header=None)
    df_ibi = pd.read_csv(os.path.join(e4_dir, "IBI.csv"), skiprows=1, header=None)
    
    eda_vals = df_eda[0].values
    temp_vals = df_temp[0].values
    hr_vals = df_hr[0].values
    ibi_times = df_ibi[0].values
    ibi_vals = df_ibi[1].values
    
    window_size = 15
    step_size = 5
    raw_results = []
    
    # 1. Extract raw features for all windows
    for t in range(0, len(hr_vals) - window_size, step_size):
        w_start = t
        w_end = t + window_size
        w_mid_min = (w_start + (window_size / 2.0)) / 60.0
        
        label = None
        for cond, (s, e) in protocol.items():
            if s <= w_mid_min <= e:
                if cond == "Base":
                    label = 1
                elif cond == "TSST":
                    label = 2
                break
                
        if label is not None:
            eda_slice = eda_vals[w_start * 4 : w_end * 4]
            temp_slice = temp_vals[w_start * 4 : w_end * 4]
            temp_slice = temp_slice[temp_slice < 50.0]
            hr_slice = hr_vals[w_start : w_end]
            
            w_start_sec = w_start
            w_end_sec = w_end
            ibi_slice = ibi_vals[(ibi_times >= w_start_sec) & (ibi_times <= w_end_sec)]
            
            if len(eda_slice) > 2 and len(temp_slice) > 2:
                # Accumulate raw metrics
                eda_mean = np.mean(eda_slice)
                eda_std = np.std(eda_slice)
                eda_min = np.min(eda_slice)
                eda_max = np.max(eda_slice)
                eda_range = eda_max - eda_min
                
                temp_mean = np.mean(temp_slice)
                temp_std = np.std(temp_slice)
                temp_min = np.min(temp_slice)
                temp_max = np.max(temp_slice)
                temp_range = temp_max - temp_min
                
                hr_mean = np.mean(hr_slice)
                hr_std = np.std(hr_slice)
                hr_min = np.min(hr_slice)
                hr_max = np.max(hr_slice)
                
                if len(ibi_slice) > 1:
                    ibi_mean = np.mean(ibi_slice)
                    ibi_sdnn = np.std(ibi_slice)
                    diff_ibi = np.diff(ibi_slice)
                    ibi_rmssd = np.sqrt(np.mean(diff_ibi ** 2)) if len(diff_ibi) > 0 else 0.0
                    ibi_pnn50 = (np.sum(np.abs(diff_ibi) > 0.05) / len(diff_ibi)) * 100 if len(diff_ibi) > 0 else 0.0
                else:
                    ibi_mean, ibi_sdnn, ibi_rmssd, ibi_pnn50 = 0.8, 0.05, 0.05, 10.0
                
                raw_results.append({
                    'minutes': w_mid_min,
                    'label': label,
                    'eda_mean': eda_mean, 'eda_std': eda_std, 'eda_min': eda_min, 'eda_max': eda_max, 'eda_range': eda_range, 'eda_slope': 0.0,
                    'temp_mean': temp_mean, 'temp_std': temp_std, 'temp_min': temp_min, 'temp_max': temp_max, 'temp_range': temp_range, 'temp_slope': 0.0,
                    'acc_mean': 0.95, 'acc_std': 0.005, 'acc_min': 0.93, 'acc_max': 0.97, # simulated
                    'hr_mean': hr_mean, 'hr_std': hr_std, 'hr_min': hr_min, 'hr_max': hr_max,
                    'ibi_mean': ibi_mean, 'ibi_sdnn': ibi_sdnn, 'ibi_rmssd': ibi_rmssd, 'ibi_pnn50': ibi_pnn50
                })
                
    if len(raw_results) == 0:
        return pd.DataFrame()
        
    session_raw_df = pd.DataFrame(raw_results)
    
    # 2. Extract baseline stats dynamically
    baseline_rows = session_raw_df[session_raw_df['label'] == 1]
    
    if len(baseline_rows) > 0:
        baseline_means = baseline_rows[FEATURE_COLS].mean()
        baseline_stds = baseline_rows[FEATURE_COLS].std().replace(0, 1.0).fillna(1.0)
    else:
        baseline_means = pd.Series({col: UNIVERSAL_BASELINE_STATS[col][0] for col in FEATURE_COLS})
        baseline_stds = pd.Series({col: UNIVERSAL_BASELINE_STATS[col][1] for col in FEATURE_COLS})
        
    # 3. Standardize and Predict
    predictions = []
    for idx, row in session_raw_df.iterrows():
        X_live = np.zeros((1, 24))
        for i, col in enumerate(FEATURE_COLS):
            mean = baseline_means[col]
            std = baseline_stds[col]
            X_live[0, i] = (row[col] - mean) / std
            
        pred = model.predict(X_live)[0]
        predictions.append(pred)
        
    session_raw_df['pred_label'] = predictions
    return session_raw_df

def main():
    st.markdown("<div class='main-header'>🧠 StressIQ: Diagnostic Center</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subheader'>Calibrated Physiological Stress Detection & Simulation</div>", unsafe_allow_html=True)
    
    # Model & Data Paths
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "stress_model_xgb.pkl")
    wesad_dir = os.path.join(script_dir, "WESAD")
    
    if not os.path.exists(model_path):
        st.error("❌ Model file `stress_model_xgb.pkl` (97.84% accurate XGBoost) not found in project folder.")
        return
        
    model = joblib.load(model_path)
    
    # 🎮 Unified Diagnostics Selector
    st.sidebar.title("🎮 Diagnostic Console")
    input_method = st.sidebar.radio(
        "Choose Input Method:",
        ["🎛️ Enter Readings Manually", "📡 Stream Smartwatch Sensors"]
    )
    
    # =========================================================================
    # METHOD 1: MANUAL READINGS CALCULATOR
    # =========================================================================
    if input_method == "🎛️ Enter Readings Manually":
        st.write("### 🎛️ Interactive Parameter Calculator")
        st.write("Adjust the sliders below to manually input physiological values:")
        
        col_inputs, col_diag = st.columns([1, 1])
        
        with col_inputs:
            input_hr = st.number_input("💓 Heart Rate (BPM)", min_value=0.0, max_value=250.0, value=75.0, step=1.0)
            input_eda = st.number_input("💧 Sweat Conductance / EDA (μS)", min_value=0.0, max_value=100.0, value=0.35, step=0.01, format="%.2f")
            input_temp = st.number_input("🌡️ Skin Temperature (°C)", min_value=0.0, max_value=60.0, value=34.5, step=0.1)
            
            st.info("ℹ️ The backend simulates a 15-second physiological window around your inputs, normalizes them using universal baseline parameters, and executes the classifier.")
            
        with col_diag:
            # Simulate physiological window centered on inputs
            np.random.seed(42)
            eda_sim = input_eda + np.random.normal(0, 0.02, 60)
            temp_sim = input_temp + np.random.normal(0, 0.05, 60)
            acc_sim = 0.95 + np.random.normal(0, 0.01, 60)
            
            mean_ibi = 60.0 / input_hr
            hrv_ratio = 0.02 if input_hr >= 85 else 0.08
            ibi_sim = mean_ibi + np.random.normal(0, mean_ibi * hrv_ratio, 15)
            hr_sim = 60.0 / ibi_sim
            
            raw_feats = {
                'eda_mean': np.mean(eda_sim),
                'eda_std': np.std(eda_sim),
                'eda_min': np.min(eda_sim),
                'eda_max': np.max(eda_sim),
                'eda_range': np.max(eda_sim) - np.min(eda_sim),
                'eda_slope': 0.0,
                
                'temp_mean': np.mean(temp_sim),
                'temp_std': np.std(temp_sim),
                'temp_min': np.min(temp_sim),
                'temp_max': np.max(temp_sim),
                'temp_range': np.max(temp_sim) - np.min(temp_sim),
                'temp_slope': 0.0,
                
                'acc_mean': np.mean(acc_sim),
                'acc_std': np.std(acc_sim),
                'acc_min': np.min(acc_sim),
                'acc_max': np.max(acc_sim),
                
                'hr_mean': np.mean(hr_sim),
                'hr_std': np.std(hr_sim),
                'hr_min': np.min(hr_sim),
                'hr_max': np.max(hr_sim),
                
                'ibi_mean': np.mean(ibi_sim),
                'ibi_sdnn': np.std(ibi_sim),
                'ibi_rmssd': np.sqrt(np.mean(np.diff(ibi_sim) ** 2)),
                'ibi_pnn50': (np.sum(np.abs(np.diff(ibi_sim)) > 0.05) / len(np.diff(ibi_sim))) * 100
            }
            
            # Normalize
            X_live = np.zeros((1, 24))
            for i, col in enumerate(FEATURE_COLS):
                mean, std = UNIVERSAL_BASELINE_STATS[col]
                X_live[0, i] = (raw_feats[col] - mean) / std
            
            # Predict
            prediction = int(model.predict(X_live)[0])
            probabilities = model.predict_proba(X_live)[0]
            confidence = probabilities[prediction]
            
            # Alert Card
            st.write("### 🚨 Diagnostic Output")
            if prediction == 1:
                st.markdown(f'<div class="pulse-stressed">🚨 DANGER: STRESSED ({confidence*100:.1f}%) 🚨</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pulse-relaxed">🟢 SYSTEM: RELAXED ({confidence*100:.1f}%)</div>', unsafe_allow_html=True)
                
            # Render HRV
            st.markdown(f"""
            <div class="card" style="margin-top: 15px;">
                <h4>⚡ Computed HRV Features (Heart Rate Variability)</h4>
                <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                    <div>
                        <h5>SDNN</h5>
                        <p style="font-size: 20px; color: #a855f7;">{raw_feats['ibi_sdnn']*1000:.1f} ms</p>
                    </div>
                    <div>
                        <h5>RMSSD</h5>
                        <p style="font-size: 20px; color: #a855f7;">{raw_feats['ibi_rmssd']*1000:.1f} ms</p>
                    </div>
                    <div>
                        <h5>pNN50</h5>
                        <p style="font-size: 20px; color: #a855f7;">{raw_feats['ibi_pnn50']:.1f}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    # =========================================================================
    # METHOD 2: STREAM SMARTWATCH SENSORS
    # =========================================================================
    else:
        st.write("### 📡 Stream Smartwatch Sensors")
        st.write("Extract and replay full physiological sessions from the WESAD database:")
        
        if not os.path.exists(wesad_dir):
            st.warning(f"WESAD Dataset directory path not found: {wesad_dir}")
            return
            
        subjects = sorted([os.path.basename(d) for d in glob.glob(os.path.join(wesad_dir, "S*")) if os.path.isdir(d)])
        selected_subject = st.selectbox("Select Subject (Simulated Wearer):", subjects)
        
        subj_dir = os.path.join(wesad_dir, selected_subject)
        quest_file = os.path.join(subj_dir, f"{selected_subject}_quest.csv")
        e4_dir = glob.glob(os.path.join(subj_dir, "*_E4*"))[0]
        
        with st.spinner("Calibrating and processing raw data..."):
            protocol = parse_quest(quest_file)
            df_analysis = run_session_analysis(e4_dir, protocol, model)
            
        if df_analysis.empty:
            st.warning("No physiological data extracted for selected subject timeline.")
            return
            
        # Stats
        total_w = len(df_analysis)
        stress_w = df_analysis['pred_label'].sum()
        stress_pct = (stress_w / total_w) * 100
        
        st.markdown(f"""
        <div class='metric-container' style="margin-top: 15px;">
            <div class='card'>
                <h3>🧠 Session Diagnostic Verdict</h3>
                <p style="color: {'#ef4444' if stress_pct > 35 else '#10b981'};">
                    { 'STRESSED' if stress_pct > 35 else 'RELAXED' }
                </p>
            </div>
            <div class='card'>
                <h3>📊 Time Stressed</h3>
                <p style="color: #3b82f6;">{stress_pct:.1f}% of Session</p>
            </div>
            <div class='card'>
                <h3>💓 Average Heart Rate</h3>
                <p style="color: #f43f5e;">{int(df_analysis['hr_mean'].mean())} BPM</p>
            </div>
            <div class='card'>
                <h3>💧 Avg Skin Conductance</h3>
                <p style="color: #06b6d4;">{df_analysis['eda_mean'].mean():.3f} μS</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Timeline chart
        st.write("### ⏱️ Session Stress Timeline")
        df_analysis['color'] = df_analysis['pred_label'].apply(lambda x: '#ef4444' if x == 1 else '#10b981')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_analysis['minutes'], y=df_analysis['hr_mean'],
            mode='lines+markers', line=dict(color='#64748b', width=1.5),
            marker=dict(color=df_analysis['color'], size=6),
            hovertemplate='Time: %{x:.2f} min<br>HR: %{y:.0f} BPM<br>Verdict: %{text}',
            text=df_analysis['pred_label'].apply(lambda x: 'STRESSED' if x == 1 else 'RELAXED')
        ))
        fig.update_layout(
            plot_bgcolor='rgba(30, 41, 59, 0.2)', paper_bgcolor='rgba(0, 0, 0, 0)',
            xaxis=dict(title="Time elapsed (minutes)", gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title="Heart Rate (BPM)", gridcolor='rgba(255,255,255,0.05)'),
            height=300, margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
