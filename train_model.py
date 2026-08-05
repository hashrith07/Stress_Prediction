import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
import joblib

def normalize_features_by_baseline(df):
    """
    Standardizes each subject's features using their baseline (label=1) mean and std.
    Converts absolute sensor readings into relative changes from their own resting baseline.
    """
    feature_cols = [
        'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range', 'eda_slope',
        'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_slope',
        'acc_mean', 'acc_std', 'acc_min', 'acc_max',
        'hr_mean', 'hr_std', 'hr_min', 'hr_max',
        'ibi_mean', 'ibi_sdnn', 'ibi_rmssd', 'ibi_pnn50'
    ]
    
    normalized_dfs = []
    for subject_id, group in df.groupby('subject_id'):
        baseline_group = group[group['label'] == 1]
        baseline_means = baseline_group[feature_cols].mean()
        baseline_stds = baseline_group[feature_cols].std()
        
        # Avoid division by zero
        baseline_stds = baseline_stds.replace(0, 1.0).fillna(1.0)
        
        group_norm = group.copy()
        for col in feature_cols:
            mean = baseline_means[col]
            std = baseline_stds[col]
            if not pd.isna(mean):
                group_norm[col] = (group[col] - mean) / std
        normalized_dfs.append(group_norm)
        
    return pd.concat(normalized_dfs, ignore_index=True)

def balance_dataset(df):
    """
    Undersamples the baseline class to match the stress class count per subject.
    """
    balanced_dfs = []
    for subject_id, group in df.groupby('subject_id'):
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
            
    return pd.concat(balanced_dfs, ignore_index=True)

def main():
    features_csv = "c:\\Users\\hashr\\OneDrive\\DS\\project\\WESAD_extracted_features.csv"
    print(f"Loading extracted features from {features_csv}...")
    df = pd.read_csv(features_csv)
    
    # Filter to Baseline (1) and Stress (2) for binary classification
    print("Filtering dataset to Baseline (1) and Stress (2)...")
    df_binary = df[df['label'].isin([1, 2])].copy()
    df_binary['binary_label'] = df_binary['label'].apply(lambda x: 1 if x == 2 else 0)
    
    # Balance classes per subject
    print("Balancing dataset (50/50 Baseline vs Stress count per subject)...")
    df_balanced = balance_dataset(df_binary)
    
    # Normalize features by subject baseline (crucial for general LOSO-CV)
    print("Normalizing features by subject-specific baseline...")
    norm_df = normalize_features_by_baseline(df_balanced)
    
    # Save the final balanced, normalized dataset to disk
    balanced_csv_path = "c:\\Users\\hashr\\OneDrive\\DS\\project\\WESAD_balanced_features.csv"
    norm_df.to_csv(balanced_csv_path, index=False)
    print(f"Saved balanced normalized dataset to: {balanced_csv_path}")
    
    # Features & target
    feature_cols = [
        'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range', 'eda_slope',
        'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_slope',
        'acc_mean', 'acc_std', 'acc_min', 'acc_max',
        'hr_mean', 'hr_std', 'hr_min', 'hr_max',
        'ibi_mean', 'ibi_sdnn', 'ibi_rmssd', 'ibi_pnn50'
    ]
    
    X = norm_df[feature_cols].values
    y = norm_df['binary_label'].values
    groups = norm_df['subject_id'].values
    
    print(f"\nFinal Dataset summary:")
    print(f"- Total samples: {len(norm_df)}")
    print(f"- Stress samples: {np.sum(y == 1)}")
    print(f"- Baseline samples: {np.sum(y == 0)}")
    print(f"- Number of subjects: {len(np.unique(groups))}")
    
    # Perform Leave-One-Subject-Out Cross-Validation (LOSO-CV)
    logo = LeaveOneGroupOut()
    
    accuracies = []
    all_y_true = []
    all_y_pred = []
    
    print("\nStarting Leave-One-Subject-Out Cross-Validation...")
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        subject_test = groups[test_idx][0]
        
        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Predict on left-out subject
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        print(f"Subject {subject_test} Left Out -> Accuracy: {acc:.4f}")
        
    mean_acc = np.mean(accuracies)
    print(f"\n======================================")
    print(f"Balanced Normalized LOSO-CV Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    print(f"======================================")
    
    # Classification Report
    print("\nClassification Report (Overall):")
    print(classification_report(all_y_true, all_y_pred, target_names=['Baseline', 'Stress']))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(all_y_true, all_y_pred))
    
    # Train the final model on the entire dataset and save it
    print("\nTraining final model on all subjects...")
    final_clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    final_clf.fit(X, y)
    
    model_output_path = "c:\\Users\\hashr\\OneDrive\\DS\\project\\stress_model.pkl"
    joblib.dump(final_clf, model_output_path)
    print(f"🎉 Final Stress Prediction model saved successfully to: {model_output_path}")

if __name__ == "__main__":
    main()
