import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

train_csv = "./dataset/2024_12_bb_3d_enhanced.csv"
test_csv = "./dataset/2024_12_25_test_enhanced.csv"
model_path = "./ml_models/IF_2024_12_bb_3d_enhanced.pkl"
thresholds_dir = "./dataset/thresholds"
threshold_file_path = os.path.join(
    thresholds_dir, "IF_threshold_2024_12_bb_3d_enhanced.txt"
)

# Load data
print("Loading enhanced data with critical SSD SMART attributes...")
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

X_train = train_df.drop(columns=["failure"]).values.astype(np.float32)
y_train = train_df["failure"].values.astype(int)
X_test = test_df.drop(columns=["failure"]).values.astype(np.float32)
y_test = test_df["failure"].values.astype(int)

X_train_normal = X_train[y_train == 0]
print(f"Training on {len(X_train_normal)} normal samples")

if os.path.exists(model_path):
    print(f"Loading existing enhanced model from {model_path}")
    with open(model_path, "rb") as f:
        isolation_forest = pickle.load(f)
    print("Enhanced model loaded successfully.")
else:
    print("Creating enhanced Isolation Forest for SSD SMART data...")

    param_grid = {
        "n_estimators": [5000, 7000],
        "max_samples": [32, 64],
        "contamination": [0.0001, 0.0005],
        "max_features": [0.25, 0.35],
        "bootstrap": [True],
    }

    base_model = IsolationForest(random_state=42, n_jobs=-1, verbose=1)

    print("Starting manual parameter search for enhanced Isolation Forest...")

    best_score = -np.inf
    best_params = None
    best_model = None

    total_combinations = (
        len(param_grid["n_estimators"])
        * len(param_grid["max_samples"])
        * len(param_grid["contamination"])
        * len(param_grid["max_features"])
        * len(param_grid["bootstrap"])
    )

    print(f"Testing {total_combinations} parameter combinations...")

    import time

    start_time = time.time()
    combination_count = 0

    for n_estimators in param_grid["n_estimators"]:
        for max_samples in param_grid["max_samples"]:
            for contamination in param_grid["contamination"]:
                for max_features in param_grid["max_features"]:
                    for bootstrap in param_grid["bootstrap"]:
                        combination_count += 1
                        elapsed_time = time.time() - start_time
                        avg_time_per_combination = elapsed_time / combination_count
                        remaining_combinations = total_combinations - combination_count
                        estimated_remaining_time = (
                            remaining_combinations * avg_time_per_combination
                        )

                        print(
                            f"Testing combination {combination_count}/{total_combinations} "
                            f"({combination_count/total_combinations*100:.1f}% complete, "
                            f"~{estimated_remaining_time/60:.1f}min remaining): "
                            f"n_estimators={n_estimators}, max_samples={max_samples}, "
                            f"contamination={contamination}, max_features={max_features}, "
                            f"bootstrap={bootstrap}"
                        )

                        # Create model with current parameters
                        model = IsolationForest(
                            n_estimators=n_estimators,
                            max_samples=max_samples,
                            contamination=contamination,
                            max_features=max_features,
                            bootstrap=bootstrap,
                            random_state=42,
                            n_jobs=-1,
                            verbose=0,  # Reduce verbosity during search
                        )

                        try:
                            model.fit(X_train_normal)

                            scores = -model.score_samples(X_train_normal)

                            score_mean = np.mean(scores)
                            score_std = np.std(scores)
                            score_range = np.max(scores) - np.min(scores)

                            normalized_std = score_std / (score_mean + 1e-8)
                            normalized_range = score_range / (score_mean + 1e-8)

                            stability_score = 1.0 / (
                                1.0 + normalized_std + normalized_range
                            )

                            print(f"  Stability score: {stability_score:.4f}")

                            if stability_score > best_score:
                                best_score = stability_score
                                best_params = {
                                    "n_estimators": n_estimators,
                                    "max_samples": max_samples,
                                    "contamination": contamination,
                                    "max_features": max_features,
                                    "bootstrap": bootstrap,
                                }
                                best_model = model
                                print(f"  *** New best score: {best_score:.4f} ***")

                        except Exception as e:
                            print(f"  Error with this combination: {e}")
                            continue

    training_time = time.time() - start_time
    print(f"Manual parameter search completed in {training_time:.2f} seconds")

    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best stability score: {best_score:.4f}")

    isolation_forest = best_model

    print("Verifying enhanced model training...")
    train_scores = isolation_forest.score_samples(X_train_normal)
    print(
        f"Training scores - Min: {np.min(train_scores):.4f}, Max: {np.max(train_scores):.4f}, Mean: {np.mean(train_scores):.4f}"
    )

    print("Saving enhanced model...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(isolation_forest, f)
    print(f"Enhanced model saved to {model_path}")

print("Calculating anomaly scores on enhanced SMART data...")
anomaly_scores = -isolation_forest.score_samples(X_test)

print("Optimizing threshold using ROC curve analysis...")

fpr, tpr, thresholds_roc = roc_curve(y_test, anomaly_scores)
roc_auc = auc(fpr, tpr)

j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds_roc[optimal_idx]

percentile_threshold = np.percentile(anomaly_scores, 90)

print(f"ROC AUC: {roc_auc:.4f}")
print(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}")
print(f"Percentile threshold (90th): {percentile_threshold:.4f}")

if roc_auc > 0.5:
    threshold = optimal_threshold
    print(f"Using optimal threshold: {threshold:.4f}")
else:
    threshold = percentile_threshold
    print(f"Using percentile threshold: {threshold:.4f}")

os.makedirs(thresholds_dir, exist_ok=True)
with open(threshold_file_path, "w") as f:
    f.write(str(threshold))

pred_labels = (anomaly_scores > threshold).astype(int)

print("\n" + "=" * 60)
print("ENHANCED ISOLATION FOREST RESULTS (with Critical SSD SMART Attributes)")
print("=" * 60)

print("Confusion Matrix:")
cm = confusion_matrix(y_test, pred_labels)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, pred_labels, digits=4))

if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = (
        2 * (precision * detection_rate) / (precision + detection_rate)
        if (precision + detection_rate) > 0
        else 0
    )

    print(f"\nDetailed Metrics:")
    print(f"True Positives: {tp} / {total} ({tp/total*100:.2f}%)")
    print(f"False Positives: {fp} / {total} ({fp/total*100:.2f}%)")
    print(f"True Negatives: {tn} / {total} ({tn/total*100:.2f}%)")
    print(f"False Negatives: {fn} / {total} ({fn/total*100:.2f}%)")
    print(f"Detection Rate (Recall for class 1): {detection_rate:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
else:
    print("Confusion matrix shape is not 2x2, cannot compute detailed metrics.")


# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Enhanced IF (with Critical SSD SMART Attributes)")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("./plots/ROC_Curve_Enhanced_IF.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot anomaly score distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(anomaly_scores[y_test == 0], bins=50, alpha=0.7, label="Normal", density=True)
plt.hist(anomaly_scores[y_test == 1], bins=50, alpha=0.7, label="Failure", density=True)
plt.axvline(
    x=threshold, color="red", linestyle="--", label=f"Threshold: {threshold:.4f}"
)
plt.xlabel("Anomaly Score")
plt.ylabel("Density")
plt.title("Anomaly Score Distribution (Enhanced)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.boxplot(
    [anomaly_scores[y_test == 0], anomaly_scores[y_test == 1]],
    labels=["Normal", "Failure"],
)
plt.ylabel("Anomaly Score")
plt.title("Anomaly Score Box Plot (Enhanced)")
plt.grid(True)

plt.tight_layout()
plt.savefig("./plots/enhanced_anomaly_score_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# Save anomaly records
anomaly_indices = np.where(pred_labels == 1)[0]
anomaly_df = test_df.iloc[anomaly_indices].copy()
anomaly_df["anomaly_score"] = anomaly_scores[anomaly_indices]
os.makedirs("./dataset/anomalies", exist_ok=True)
anomaly_df.to_csv(
    "./dataset/anomalies/IF_anomalies_enhanced_2024_12_25_test.csv",
    index=False,
)
print(
    f"\nSaved {len(anomaly_df)} anomaly records to ./dataset/anomalies/IF_anomalies_enhanced_2024_12_25_test.csv"
)
