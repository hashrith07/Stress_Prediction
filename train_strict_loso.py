import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
import xgboost as xgb

def balance_training_data(train_df):
    """
    Balances the training data by downsampling the majority class (Baseline)
    to match the minority class (Stress) count per subject.
    """
    balanced_dfs = []
    for subject_id, group in train_df.groupby('subject_id'):
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

def normalize_training_data_calibrated(train_df, feature_cols):
    """
    Normalization for Calibrated Scenario:
    Standardizes each training subject's features using their own baseline (label=1) mean and std.
    """
    normalized_dfs = []
    for subject_id, group in train_df.groupby('subject_id'):
        baseline_group = group[group['label'] == 1]
        baseline_means = baseline_group[feature_cols].mean()
        baseline_stds = baseline_group[feature_cols].std().replace(0, 1.0).fillna(1.0)
        
        group_norm = group.copy()
        for col in feature_cols:
            mean = baseline_means[col]
            std = baseline_stds[col]
            if not pd.isna(mean):
                group_norm[col] = (group[col] - mean) / std
        normalized_dfs.append(group_norm)
        
    return pd.concat(normalized_dfs, ignore_index=True)

def main():
    features_csv = "c:\\Users\\hashr\\OneDrive\\DS\\project\\WESAD_extracted_features.csv"
    print(f"Loading raw extracted features from {features_csv}...")
    df = pd.read_csv(features_csv)
    
    # Filter to Baseline (1) and Stress (2) for binary classification
    df_binary = df[df['label'].isin([1, 2])].copy()
    df_binary['binary_label'] = df_binary['label'].apply(lambda x: 1 if x == 2 else 0)
    
    feature_cols = [
        'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range', 'eda_slope',
        'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_slope',
        'acc_mean', 'acc_std', 'acc_min', 'acc_max',
        'hr_mean', 'hr_std', 'hr_min', 'hr_max',
        'ibi_mean', 'ibi_sdnn', 'ibi_rmssd', 'ibi_pnn50'
    ]
    
    subjects = df_binary['subject_id'].unique()
    
    # Track metrics for each subject
    sub_metrics_calib = []
    sub_metrics_uncalib = []
    
    all_y_true = []
    all_y_pred_calib = []
    all_y_pred_uncalib = []
    all_y_prob_calib = []
    all_y_prob_uncalib = []
    
    # Accumulate feature importances across folds
    feature_importances = np.zeros(len(feature_cols))
    
    print("\nStarting Leak-Proof Leave-One-Subject-Out Cross-Validation with Hyperparameter Tuning...")
    print("-" * 110)
    print(f"{'Subject':<10} | {'Calib Acc':<11} | {'Calib F1':<11} | {'Calib AUC':<11} | {'Uncal Acc':<11} | {'Uncal F1':<11} | {'Uncal AUC':<11}")
    print("-" * 110)
    
    for test_subj in subjects:
        # 1. Split train/test subjects
        train_df = df_binary[df_binary['subject_id'] != test_subj].copy()
        test_df = df_binary[df_binary['subject_id'] == test_subj].copy()
        
        # 2. Balance ONLY the training subjects
        train_df_balanced = balance_training_data(train_df)
        
        # 3. Create normalizations and train classifiers
        y_test = test_df['binary_label'].values
        
        # =====================================================================
        # SCENARIO A: CALIBRATED DEPLOYMENT
        # (Assumes a new user provides a baseline recording before predictions)
        # =====================================================================
        train_df_calib_norm = normalize_training_data_calibrated(train_df_balanced, feature_cols)
        X_train_calib = train_df_calib_norm[feature_cols].values
        y_train_calib = train_df_calib_norm['binary_label'].values
        groups_train_calib = train_df_calib_norm['subject_id'].values
        
        # Personal baseline normalization of test subject
        test_baseline = test_df[test_df['label'] == 1]
        test_baseline_means = test_baseline[feature_cols].mean()
        test_baseline_stds = test_baseline[feature_cols].std().replace(0, 1.0).fillna(1.0)
        
        test_df_calib = test_df.copy()
        for col in feature_cols:
            mean = test_baseline_means[col]
            std = test_baseline_stds[col]
            if not pd.isna(mean):
                test_df_calib[col] = (test_df[col] - mean) / std
        X_test_calib = test_df_calib[feature_cols].values
        
        # Inner-fold Hyperparameter tuning to prevent leakage
        gkf = GroupKFold(n_splits=3)
        param_grid = {
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1]
        }
        grid_calib = GridSearchCV(
            estimator=xgb.XGBClassifier(
                objective="binary:logistic",
                n_estimators=100,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42
            ),
            param_grid=param_grid,
            cv=gkf,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_calib.fit(X_train_calib, y_train_calib, groups=groups_train_calib)
        clf_calib = grid_calib.best_estimator_
        
        # Predict
        y_pred_calib = clf_calib.predict(X_test_calib)
        y_prob_calib = clf_calib.predict_proba(X_test_calib)[:, 1]
        
        # Feature importance accumulation
        feature_importances += clf_calib.feature_importances_ / len(subjects)
        
        # Compute metrics
        acc_c = accuracy_score(y_test, y_pred_calib)
        f1_c = f1_score(y_test, y_pred_calib)
        auc_c = roc_auc_score(y_test, y_prob_calib) if len(np.unique(y_test)) > 1 else np.nan
        sub_metrics_calib.append((acc_c, f1_c, auc_c))
        
        # =====================================================================
        # SCENARIO B: UNCALIBRATED DEPLOYMENT
        # (Assumes no personal baseline is available - uses training-derived norm)
        # =====================================================================
        # Normalize BOTH training and testing using the same global training statistics
        train_raw_means = train_df_balanced[feature_cols].mean()
        train_raw_stds = train_df_balanced[feature_cols].std().replace(0, 1.0).fillna(1.0)
        
        # Normalize training fold globally
        X_train_uncalib = (train_df_balanced[feature_cols].values - train_raw_means.values) / train_raw_stds.values
        y_train_uncalib = train_df_balanced['binary_label'].values
        groups_train_uncalib = train_df_balanced['subject_id'].values
        
        # Normalize test subject globally using same training averages
        X_test_uncalib = (test_df[feature_cols].values - train_raw_means.values) / train_raw_stds.values
        
        # Inner-fold Hyperparameter tuning
        grid_uncalib = GridSearchCV(
            estimator=xgb.XGBClassifier(
                objective="binary:logistic",
                n_estimators=100,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42
            ),
            param_grid=param_grid,
            cv=gkf,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_uncalib.fit(X_train_uncalib, y_train_uncalib, groups=groups_train_uncalib)
        clf_uncalib = grid_uncalib.best_estimator_
        
        # Predict
        y_pred_uncalib = clf_uncalib.predict(X_test_uncalib)
        y_prob_uncalib = clf_uncalib.predict_proba(X_test_uncalib)[:, 1]
        
        # Compute metrics
        acc_u = accuracy_score(y_test, y_pred_uncalib)
        f1_u = f1_score(y_test, y_pred_uncalib)
        auc_u = roc_auc_score(y_test, y_prob_uncalib) if len(np.unique(y_test)) > 1 else np.nan
        sub_metrics_uncalib.append((acc_u, f1_u, auc_u))
        
        # Accumulate overall stats
        all_y_true.extend(y_test)
        all_y_pred_calib.extend(y_pred_calib)
        all_y_pred_uncalib.extend(y_pred_uncalib)
        all_y_prob_calib.extend(y_prob_calib)
        all_y_prob_uncalib.extend(y_prob_uncalib)
        
        print(f"{test_subj:<10} | {acc_c*100:<9.2f}% | {f1_c*100:<9.2f}% | {auc_c*100:<9.2f}% | {acc_u*100:<9.2f}% | {f1_u*100:<9.2f}% | {auc_u*100:<9.2f}%")
        
    # Calculate statistics (Mean +/- SD)
    accs_c, f1s_c, aucs_c = zip(*sub_metrics_calib)
    accs_u, f1s_u, aucs_u = zip(*sub_metrics_uncalib)
    
    print("-" * 110)
    print(f"{'Mean':<10} | {np.mean(accs_c)*100:<9.2f}% | {np.mean(f1s_c)*100:<9.2f}% | {np.mean(aucs_c)*100:<9.2f}% | {np.mean(accs_u)*100:<9.2f}% | {np.mean(f1s_u)*100:<9.2f}% | {np.mean(aucs_u)*100:<9.2f}%")
    print(f"{'Std Dev':<10} | {np.std(accs_c)*100:<9.2f}% | {np.std(f1s_c)*100:<9.2f}% | {np.std(aucs_c)*100:<9.2f}% | {np.std(accs_u)*100:<9.2f}% | {np.std(f1s_u)*100:<9.2f}% | {np.std(aucs_u)*100:<9.2f}%")
    print("-" * 110)
    
    # Overall evaluations
    print("\n" + "="*80)
    print("🏆 SUMMARY REPORT: CALIBRATED DEPLOYMENT SCENARIO")
    print("Description: Assumes a new user provides a baseline recording before predictions.")
    print("="*80)
    print(classification_report(all_y_true, all_y_pred_calib, target_names=['Baseline', 'Stress']))
    print(f"Overall ROC-AUC Score: {roc_auc_score(all_y_true, all_y_prob_calib):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_y_true, all_y_pred_calib))
    
    print("\n" + "="*80)
    print("🏆 SUMMARY REPORT: UNCALIBRATED DEPLOYMENT SCENARIO")
    print("Description: Assumes no personal baseline is available - uses training-derived norm.")
    print("="*80)
    print(classification_report(all_y_true, all_y_pred_uncalib, target_names=['Baseline', 'Stress']))
    print(f"Overall ROC-AUC Score: {roc_auc_score(all_y_true, all_y_prob_uncalib):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_y_true, all_y_pred_uncalib))
    
    # Feature Importance Analysis
    print("\n" + "="*80)
    print("🏆 FEATURE IMPORTANCE ANALYSIS (XGBoost average across folds)")
    print("="*80)
    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    for idx, row in fi_df.iterrows():
        print(f"{row['Feature']:<25} : {row['Importance']:.4f}")

if __name__ == "__main__":
    main()
