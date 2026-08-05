import pandas as pd
import xgboost as xgb
import joblib

def main():
    features_csv = "c:\\Users\\hashr\\OneDrive\\DS\\project\\WESAD_balanced_features.csv"
    print(f"Loading balanced features from {features_csv}...")
    df = pd.read_csv(features_csv)
    
    # Feature columns
    feature_cols = [
        'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range', 'eda_slope',
        'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_slope',
        'acc_mean', 'acc_std', 'acc_min', 'acc_max',
        'hr_mean', 'hr_std', 'hr_min', 'hr_max',
        'ibi_mean', 'ibi_sdnn', 'ibi_rmssd', 'ibi_pnn50'
    ]
    
    X = df[feature_cols].values
    y = df['binary_label'].values
    
    print("\nTraining final XGBoost Classifier on all subjects...")
    # Train the final XGBoost model
    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    clf.fit(X, y)
    
    model_output_path = "c:\\Users\\hashr\\OneDrive\\DS\\project\\stress_model_xgb.pkl"
    joblib.dump(clf, model_output_path)
    print(f"🎉 Final XGBoost Stress Prediction model saved successfully to: {model_output_path}")

if __name__ == "__main__":
    main()
