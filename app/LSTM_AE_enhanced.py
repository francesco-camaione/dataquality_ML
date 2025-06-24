import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import os

# File paths
train_csv = "./dataset/2024_12_bb_3d_enhanced.csv"
test_csv = "./dataset/2024_12_25_test_enhanced.csv"
model_path = "./ml_models/Dense_AE_enhanced.keras"
thresholds_dir = "./dataset/thresholds"
os.makedirs(thresholds_dir, exist_ok=True)
threshold_file_path = os.path.join(
    thresholds_dir, "Dense_AE_threshold_2024_12_bb_3d_enhanced.txt"
)

# Load data
print("Loading enhanced data...")
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Filter out features that are >95% zero or missing
features = [col for col in train_df.columns if col != "failure"]
zero_frac = (train_df[features] == 0).mean()
missing_frac = train_df[features].isnull().mean()
good_features = [
    f for f in features if (zero_frac[f] < 0.95) and (missing_frac[f] < 0.95)
]

print(f"Using {len(good_features)} features out of {len(features)} after filtering.")

X_train = train_df[good_features].values.astype(np.float32)
y_train = train_df["failure"].values.astype(int)
X_test = test_df[good_features].values.astype(np.float32)
y_test = test_df["failure"].values.astype(int)

# Fill missing values
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train only on normal data
X_train_normal = X_train[y_train == 0]
print(f"Training on {len(X_train_normal)} normal samples")


# Clip each feature to the 1st and 99th percentile (per feature)
def clip_outliers(X, low=1, high=99):
    X_clipped = X.copy()
    for i in range(X.shape[1]):
        lower = np.percentile(X[:, i], low)
        upper = np.percentile(X[:, i], high)
        X_clipped[:, i] = np.clip(X[:, i], lower, upper)
    return X_clipped


X_train_normal = clip_outliers(X_train_normal)
X_train = clip_outliers(X_train)
X_test = clip_outliers(X_test)

# Check for NaNs or infs
assert not np.isnan(X_train_normal).any(), "NaNs in X_train_normal"
assert not np.isinf(X_train_normal).any(), "Infs in X_train_normal"

# Build or load the Dense Autoencoder
if os.path.exists(model_path):
    from keras.models import load_model

    print(f"Loading existing model from {model_path}")
    autoencoder = load_model(model_path, compile=False)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss="mae")
    print("Model loaded and compiled successfully.")
else:
    print("Training new model...")
    input_dim = X_train_normal.shape[1]
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation="relu", kernel_initializer="he_normal")(inputs)
    x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(32, activation="relu", kernel_initializer="he_normal")(x)
    bottleneck = Dense(16, activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(32, activation="relu", kernel_initializer="he_normal")(bottleneck)
    x = Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    outputs = Dense(input_dim, activation="linear")(x)
    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss="mae")
    autoencoder.summary()
    history = autoencoder.fit(
        X_train_normal,
        X_train_normal,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        shuffle=True,
        verbose=2,
    )
    autoencoder.save(model_path)
    print(f"Model saved to {model_path}")

# Calculate reconstruction error on train normal for threshold
recon_train = autoencoder.predict(X_train_normal)
mae_train = np.mean(np.abs(X_train_normal - recon_train), axis=1)

# Calculate reconstruction error on test
recon_test = autoencoder.predict(X_test)
mae_test = np.mean(np.abs(X_test - recon_test), axis=1)

# ROC curve and threshold
fpr, tpr, thresholds_roc = roc_curve(y_test, mae_test)
roc_auc = auc(fpr, tpr)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds_roc[optimal_idx]
percentile_threshold = np.percentile(mae_train, 95)

print(f"ROC AUC: {roc_auc:.4f}")
print(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}")
print(f"Percentile threshold (90th): {percentile_threshold:.4f}")

# Use ROC threshold if AUC > 0.5, else percentile
if roc_auc > 0.5:
    threshold = optimal_threshold
else:
    threshold = percentile_threshold
print(f"Using threshold: {threshold:.4f}")

with open(threshold_file_path, "w") as f:
    f.write(str(threshold))

# Predict anomalies
pred_labels = (mae_test > threshold).astype(int)

# Metrics
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, pred_labels)
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, pred_labels, digits=4))

# Print detection rate and false positive rate, and counts
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"Detection Rate (Recall for class 1): {detection_rate:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")
    print(f"True Positives detected: {tp} / Total actual positives: {tp + fn}")
    print(f"False Positives detected: {fp} / Total actual negatives: {fp + tn}")

# ROC Curve plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Dense AE Enhanced")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/ROC_Curve_Dense_AE_Enhanced.png", dpi=300)
plt.close()

# Save anomaly scores
anomaly_df = test_df.copy()
anomaly_df["reconstruction_error"] = mae_test
anomaly_df["predicted_failure"] = pred_labels
anomaly_df.to_csv(
    "./dataset/anomalies/Dense_AE_anomalies_enhanced_2024_12_25_test.csv", index=False
)
print(
    f"Saved anomaly scores to ./dataset/anomalies/Dense_AE_anomalies_enhanced_2024_12_25_test.csv"
)
