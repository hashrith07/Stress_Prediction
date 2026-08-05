import pandas as pd

def main():
    raw_features_path = "c:\\Users\\hashr\\OneDrive\\DS\\project\\WESAD_extracted_features.csv"
    balanced_raw_path = "c:\\Users\\hashr\\OneDrive\\DS\\project\\WESAD_balanced_raw_features.csv"
    
    print(f"Loading raw features from {raw_features_path}...")
    df = pd.read_csv(raw_features_path)
    
    # Filter to Baseline (1) and Stress (2) for binary classification
    print("Filtering dataset to Baseline (1) and Stress (2)...")
    df_binary = df[df['label'].isin([1, 2])].copy()
    
    # Ensure binary_label is present
    df_binary['binary_label'] = df_binary['label'].apply(lambda x: 1 if x == 2 else 0)
    
    print("Balancing classes (50/50 Baseline vs Stress count per subject)...")
    balanced_dfs = []
    
    for subject_id, group in df_binary.groupby('subject_id'):
        stress_group = group[group['binary_label'] == 1]
        baseline_group = group[group['binary_label'] == 0]
        
        n_stress = len(stress_group)
        n_baseline = len(baseline_group)
        target_n = min(n_stress, n_baseline)
        
        if target_n > 0:
            stress_sampled = stress_group.sample(n=target_n, random_state=42)
            baseline_sampled = baseline_group.sample(n=target_n, random_state=42)
            balanced_dfs.append(stress_sampled)
            balanced_dfs.append(baseline_sampled)
            
    # Combine back
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Save to disk
    balanced_df.to_csv(balanced_raw_path, index=False)
    
    print(f"\n🎉 Successfully created class-balanced raw dataset!")
    print(f"- Saved to: {balanced_raw_path}")
    print(f"- Total rows: {len(balanced_df)}")
    print(f"- Stress rows: {len(balanced_df[balanced_df['binary_label'] == 1])}")
    print(f"- Baseline rows: {len(balanced_df[balanced_df['binary_label'] == 0])}")

if __name__ == "__main__":
    main()
