import os
import pandas as pd
import numpy as np
from scipy.stats import linregress

def detect_r_peaks(ecg_signal, fs=700):
    """
    Robust threshold-based ECG R-peak detection.
    """
    mean_val = np.mean(ecg_signal)
    std_val = np.std(ecg_signal) + 1e-6
    ecg_norm = (ecg_signal - mean_val) / std_val
    
    # R-peaks are massive positive deflections (typically > 2.5 standard deviations)
    threshold = 2.5
    peaks = np.where(ecg_norm > threshold)[0]
    
    # Human refractory period: heartbeats must be at least 0.4 seconds apart (280 samples at 700Hz)
    min_dist = int(0.4 * fs)
    filtered_peaks = []
    if len(peaks) > 0:
        last_peak = peaks[0]
        filtered_peaks.append(last_peak)
        for p in peaks[1:]:
            if p - last_peak >= min_dist:
                filtered_peaks.append(p)
                last_peak = p
                
    return np.array(filtered_peaks)

def extract_features_from_window(eda_slice, temp_slice, acc_mag_slice, ecg_slice, label, subject_id, fs=700):
    """
    Extracts statistical and physiological features from a 15-second window slice.
    """
    # 1. EDA features
    eda_mean = np.mean(eda_slice)
    eda_std = np.std(eda_slice) if len(eda_slice) > 1 else 0.0
    eda_min = np.min(eda_slice)
    eda_max = np.max(eda_slice)
    eda_range = eda_max - eda_min
    eda_slope = linregress(np.arange(len(eda_slice)), eda_slice).slope if len(eda_slice) > 1 else 0.0
    
    # 2. TEMP features
    temp_mean = np.mean(temp_slice)
    temp_std = np.std(temp_slice) if len(temp_slice) > 1 else 0.0
    temp_min = np.min(temp_slice)
    temp_max = np.max(temp_slice)
    temp_range = temp_max - temp_min
    temp_slope = linregress(np.arange(len(temp_slice)), temp_slice).slope if len(temp_slice) > 1 else 0.0
    
    # 3. ACC features
    acc_mean = np.mean(acc_mag_slice)
    acc_std = np.std(acc_mag_slice) if len(acc_mag_slice) > 1 else 0.0
    acc_min = np.min(acc_mag_slice)
    acc_max = np.max(acc_mag_slice)
    
    # 4. ECG/HRV features
    peaks = detect_r_peaks(ecg_slice, fs)
    
    if len(peaks) > 1:
        # Inter-Beat Intervals (IBIs) in seconds
        ibi_slice = np.diff(peaks) / fs
        
        ibi_mean = np.mean(ibi_slice)
        ibi_sdnn = np.std(ibi_slice)
        diff_ibi = np.diff(ibi_slice)
        ibi_rmssd = np.sqrt(np.mean(diff_ibi ** 2)) if len(diff_ibi) > 0 else 0.0
        ibi_pnn50 = (np.sum(np.abs(diff_ibi) > 0.05) / len(diff_ibi)) * 100 if len(diff_ibi) > 0 else 0.0
        
        # Calculate Heart Rate (BPM)
        hr_mean = 60.0 / ibi_mean
        hr_std = np.std(60.0 / ibi_slice) if len(ibi_slice) > 1 else 0.0
        hr_min = np.min(60.0 / ibi_slice)
        hr_max = np.max(60.0 / ibi_slice)
    else:
        # Fallbacks if peak detection fails due to noise
        ibi_mean = np.nan
        ibi_sdnn = np.nan
        ibi_rmssd = np.nan
        ibi_pnn50 = np.nan
        hr_mean = np.nan
        hr_std = np.nan
        hr_min = np.nan
        hr_max = np.nan
        
    return {
        'subject_id': subject_id,
        'label': label,
        'eda_mean': eda_mean,
        'eda_std': eda_std,
        'eda_min': eda_min,
        'eda_max': eda_max,
        'eda_range': eda_range,
        'eda_slope': eda_slope,
        'temp_mean': temp_mean,
        'temp_std': temp_std,
        'temp_min': temp_min,
        'temp_max': temp_max,
        'temp_range': temp_range,
        'temp_slope': temp_slope,
        'acc_mean': acc_mean,
        'acc_std': acc_std,
        'acc_min': acc_min,
        'acc_max': acc_max,
        'hr_mean': hr_mean,
        'hr_std': hr_std,
        'hr_min': hr_min,
        'hr_max': hr_max,
        'ibi_mean': ibi_mean,
        'ibi_sdnn': ibi_sdnn,
        'ibi_rmssd': ibi_rmssd,
        'ibi_pnn50': ibi_pnn50
    }

def main():
    csv_path = "c:\\Users\\hashr\\OneDrive\\DS\\project\\wesad_100percent.csv"
    output_path = "c:\\Users\\hashr\\OneDrive\\DS\\project\\WESAD_extracted_features.csv"
    
    print(f"Opening 9GB dataset: {csv_path}")
    
    # 700Hz Parameters
    fs = 700
    window_sec = 15
    step_sec = 1 # Slide window every 1 second for smooth real-time telemetry
    
    window_samples = window_sec * fs  # 10,500 samples
    step_samples = step_sec * fs      # 700 samples
    
    # We read in chunks of 5 million rows
    chunk_size = 5000000
    features_list = []
    
    # Active states mapping: 1=Baseline, 2=Stress, 3=Amusement
    active_labels = [1, 2, 3]
    
    # Track excess rows at chunk boundaries
    overflow_df = pd.DataFrame()
    
    print("\nProcessing streaming chunks...")
    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        print(f"Processing Chunk {chunk_idx + 1}...")
        
        # Combine overflow from previous chunk
        if not overflow_df.empty:
            chunk = pd.concat([overflow_df, chunk], ignore_index=True)
            
        # Group by Subject to ensure boundaries
        for subject_id, group in chunk.groupby('Subject'):
            # Precompute accelerometer magnitude
            acc_mag = np.sqrt(group['ACC_X']**2 + group['ACC_Y']**2 + group['ACC_Z']**2).values
            eda_vals = group['EDA'].values
            temp_vals = group['TEMP'].values
            ecg_vals = group['ECG'].values
            labels = group['Label'].values
            
            n_samples = len(group)
            
            # Slide window
            for idx in range(0, n_samples - window_samples, step_samples):
                w_start = idx
                w_end = idx + window_samples
                
                # We label the window using the majority label in the slice
                window_labels = labels[w_start:w_end]
                majority_label = int(pd.Series(window_labels).mode()[0])
                
                if majority_label in active_labels:
                    feat = extract_features_from_window(
                        eda_slice=eda_vals[w_start:w_end],
                        temp_slice=temp_vals[w_start:w_end],
                        acc_mag_slice=acc_mag[w_start:w_end],
                        ecg_slice=ecg_vals[w_start:w_end],
                        label=majority_label,
                        subject_id=subject_id,
                        fs=fs
                    )
                    features_list.append(feat)
                    
            # Keep track of the last partial window for the next chunk
            excess_idx = (n_samples // step_samples) * step_samples
            if excess_idx < n_samples:
                overflow_df = group.iloc[excess_idx:]
            else:
                overflow_df = pd.DataFrame()
                
        # Limit total feature size for performance
        if len(features_list) >= 60000:
            print("Reached target feature size limit of 60,000 samples. Saving...")
            break
            
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Fill any NaN heart rates using interpolation
    features_df = features_df.interpolate(method='linear').ffill().bfill()
    
    # Add binary_label column (1=Stress, 0=Non-Stress)
    features_df['binary_label'] = features_df['label'].apply(lambda x: 1 if x == 2 else 0)
    
    # Save raw extracted features to disk
    features_df.to_csv(output_path, index=False)
    print(f"\n🎉 Feature Extraction Complete! Extracted {len(features_df)} smartwatch rows.")
    print(f"Saved dataset to: {output_path}")

if __name__ == "__main__":
    main()
