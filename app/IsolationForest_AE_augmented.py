import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# File paths
train_csv = "./dataset/2024_12_bb_3d_augmented.csv"
test_csv = "./dataset/2024_12_25_test_augmented.csv"
model_path = "./ml_models/IF_2024_12_bb_3d_augmented.pkl"

# Load data
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Use all columns except 'failure' as features
X_train = train_df.drop(columns=["failure"]).values.astype(np.float32)
X_test = test_df.drop(columns=["failure"]).values.astype(np.float32)
y_test = test_df["failure"].values.astype(int)

# Train or load model
if os.path.exists(model_path):
    print(f"Loading existing Isolation Forest model from {model_path}")
    isolation_forest = joblib.load(model_path)
else:
    print("Training Isolation Forest model...")
    isolation_forest = IsolationForest(
        n_estimators=100, contamination="auto", random_state=42, n_jobs=-1
    )
    isolation_forest.fit(X_train)
    os.makedirs("./ml_models", exist_ok=True)
    joblib.dump(isolation_forest, model_path)
    print(f"Model saved to {model_path}")

# Predict anomaly scores (lower = more anomalous)
anomaly_scores = -isolation_forest.decision_function(X_test)
# Use 95th percentile as threshold (top 5% most anomalous are flagged)
threshold = np.percentile(anomaly_scores, 85)
pred_labels = (anomaly_scores > threshold).astype(
    int
)  # 1 = anomaly (failure), 0 = normal

# Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_labels))
print("Classification Report:")
print(classification_report(y_test, pred_labels, digits=4))

cm = confusion_matrix(y_test, pred_labels)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"True Positives: {tp} / {total}")
    print(f"False Positives: {fp} / {total}")
    print(f"Detection Rate (Recall for class 1): {detection_rate:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")
else:
    print(
        "Confusion matrix shape is not 2x2, cannot compute detection/false positive rates."
    )

# Save anomaly records
anomaly_indices = np.where(pred_labels == 1)[0]
anomaly_df = test_df.iloc[anomaly_indices].copy()
anomaly_df["anomaly_score"] = anomaly_scores[anomaly_indices]
os.makedirs("./dataset/anomalies", exist_ok=True)
anomaly_df.to_csv(
    "./dataset/anomalies/IF_anomalies_2024_12_25_test_augmented.csv", index=False
)
print(
    f"Saved {len(anomaly_df)} anomaly records to ./dataset/anomalies/IF_anomalies_2024_12_25_test_augmented.csv"
)
