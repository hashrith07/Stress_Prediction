import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
import xgboost as xgb

def evaluate_loso(clf, X, y, groups):
    """
    Evaluates a classifier using Leave-One-Subject-Out Cross-Validation.
    """
    logo = LeaveOneGroupOut()
    accuracies = []
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
    return np.mean(accuracies)

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
    groups = df['subject_id'].values
    
    print("\nDataset Summary:")
    print(f"- Total rows: {len(df)}")
    print(f"- Subjects: {len(np.unique(groups))}")
    
    # Initialize classifiers
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')
    }
    
    results = {}
    
    print("\nEvaluating models using Leave-One-Subject-Out (LOSO) Cross-Validation...")
    for name, clf in models.items():
        print(f"Training and evaluating {name}...")
        mean_acc = evaluate_loso(clf, X, y, groups)
        results[name] = mean_acc
        print(f"-> {name} LOSO Accuracy: {mean_acc*100:.2f}%")
        
    # Render final comparison table
    print("\n==========================================")
    print("🏆 MACHINE LEARNING MODEL COMPARISON (LOSO)")
    print("==========================================")
    print(f"{'Classifier':<25} | {'LOSO Accuracy':<15}")
    print("-" * 45)
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<25} | {score*100:.2f}%")
    print("==========================================")

if __name__ == "__main__":
    main()
