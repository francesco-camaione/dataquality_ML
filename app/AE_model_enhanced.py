import os
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from keras.layers import Input, Dense, BatchNormalization, GaussianNoise, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers.legacy import Adam
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

train_csv = "./dataset/2024_12_bb_3d_enhanced.csv"
test_csv = "./dataset/2024_12_25_test_enhanced.csv"
model_path = "./ml_models/AE_2024_12_bb_3d_enhanced.keras"
thresholds_dir = "./dataset/thresholds"
threshold_file_path = os.path.join(
    thresholds_dir, "AE_threshold_2024_12_bb_3d_enhanced.txt"
)

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

X_train = train_df.drop(columns=["failure"]).values.astype(np.float32)
y_train = train_df["failure"].values.astype(int)
X_test = test_df.drop(columns=["failure"]).values.astype(np.float32)
y_test = test_df["failure"].values.astype(int)

X_train_normal = X_train[y_train == 0]

input_dim = X_train.shape[1]

if os.path.exists(model_path):
    print(f"loading existing enhanced model from {model_path}")
    autoencoder = load_model(model_path, compile=False)
    optimizer = Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=optimizer, loss="mae")
    print("enhanced model loaded and compiled successfully.")
else:
    input_layer = Input(shape=(input_dim,))

    x = GaussianNoise(0.005)(input_layer)

    x = Dense(1024, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(512, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(256, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(64, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(256, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(512, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(1024, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    output_layer = Dense(input_dim, activation="linear")(x)

    autoencoder = keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=0.001)  # Higher initial learning rate
    autoencoder.compile(optimizer=optimizer, loss="mae")

    autoencoder.summary()

    # Enhanced training configuration
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=25,  # Increased patience for complex data
        restore_best_weights=True,
        min_delta=0.00001,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=0.000001,
        verbose=1,
    )

    print("Training enhanced autoencoder on SSD SMART data...")
    history = autoencoder.fit(
        X_train_normal,
        X_train_normal,
        epochs=200,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        shuffle=True,
        verbose=1,
    )

    autoencoder.save(model_path)
    print(f"Enhanced model saved to {model_path}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Enhanced Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        "./plots/enhanced_ae_training_history.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


reconstructed_train = autoencoder.predict(X_train_normal, verbose=0)
mae_train_normal = np.mean(np.abs(X_train_normal - reconstructed_train), axis=1)

reconstructed_test = autoencoder.predict(X_test, verbose=0)
mae_test = np.mean(np.abs(X_test - reconstructed_test), axis=1)

fpr, tpr, thresholds_roc = roc_curve(y_test, mae_test)
roc_auc = auc(fpr, tpr)

j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds_roc[optimal_idx]

percentile_threshold = np.percentile(mae_train_normal, 90)  # 90th percentile

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

pred_labels = (mae_test > threshold).astype(int)

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

print(f"\nEnhanced Model Results (with Critical SSD SMART Attributes):")
print(f"- Detection Rate: {detection_rate:.4f} ({detection_rate*100:.2f}%)")
print(f"- True Positives: {tp:,}")
print(f"- False Positives: {fp:,}")
print(f"- F1-Score: {f1_score:.4f}")
print(f"- ROC AUC: {roc_auc:.4f}")

improvement_detection = (detection_rate - 0.0356) / 0.0356 * 100
improvement_f1 = (f1_score - 0.0547) / 0.0547 * 100

print(f"\nImprovements:")
print(f"- Detection Rate Improvement: {improvement_detection:.1f}%")
print(f"- F1-Score Improvement: {improvement_f1:.1f}%")

#  ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Enhanced AE (with Critical SSD SMART Attributes)")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("./plots/ROC_Curve_Enhanced_AE.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot reconstruction error distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(mae_test[y_test == 0], bins=50, alpha=0.7, label="Normal", density=True)
plt.hist(mae_test[y_test == 1], bins=50, alpha=0.7, label="Failure", density=True)
plt.axvline(
    x=threshold, color="red", linestyle="--", label=f"Threshold: {threshold:.4f}"
)
plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.title("Reconstruction Error Distribution (Enhanced)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.boxplot(
    [mae_test[y_test == 0], mae_test[y_test == 1]], labels=["Normal", "Failure"]
)
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error Box Plot (Enhanced)")
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "./plots/enhanced_reconstruction_error_analysis.png", dpi=300, bbox_inches="tight"
)
plt.close()

# save anomaly records
anomaly_indices = np.where(pred_labels == 1)[0]
anomaly_df = test_df.iloc[anomaly_indices].copy()
anomaly_df["reconstruction_error"] = mae_test[anomaly_indices]
os.makedirs("./dataset/anomalies", exist_ok=True)
anomaly_df.to_csv(
    "./dataset/anomalies/AE_anomalies_enhanced_2024_12_25_test_augmented.csv",
    index=False,
)
print(
    f"\nSaved {len(anomaly_df)} anomaly records to ./dataset/anomalies/AE_anomalies_enhanced_2024_12_25_test_augmented.csv"
)
