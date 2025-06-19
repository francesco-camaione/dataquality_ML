import os
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from keras.layers import Input, Dense, BatchNormalization, Dropout, GaussianNoise
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers.legacy import Adam
from sklearn.metrics import roc_curve, auc

# File paths
train_csv = "./dataset/2024_12_bb_3d_augmented.csv"
test_csv = "./dataset/2024_12_25_test_augmented.csv"
model_path = "./ml_models/AE_2024_12_bb_3d_augmented.keras"
thresholds_dir = "./dataset/thresholds"
threshold_file_path = os.path.join(
    thresholds_dir, "AE_threshold_2024_12_bb_3d_augmented.txt"
)

# Load data
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Split features and labels
X_train = train_df.drop(columns=["failure"]).values.astype(np.float32)
y_train = train_df["failure"].values.astype(int)
X_test = test_df.drop(columns=["failure"]).values.astype(np.float32)
y_test = test_df["failure"].values.astype(int)

# Train only on normal data
X_train_normal = X_train[y_train == 0]

# Model definition (same as AE_model.py)
input_dim = X_train.shape[1]

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    autoencoder = load_model(model_path, compile=False)
    optimizer = Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=optimizer, loss="mae")
    print("Model loaded and compiled successfully.")
else:
    input_layer = Input(shape=(input_dim,))
    x = GaussianNoise(0.01)(input_layer)
    x = Dense(512, activation="relu", kernel_regularizer=l2(0.00025))(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(0.00025))(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.00025))(x)
    x = BatchNormalization()(x)
    x = Dense(48, activation="relu", kernel_regularizer=l2(0.00025))(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="linear", kernel_regularizer=l2(0.00025))(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="linear", kernel_regularizer=l2(0.00025))(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="linear", kernel_regularizer=l2(0.00025))(x)
    x = BatchNormalization()(x)
    output_layer = Dense(input_dim, activation="linear")(x)

    autoencoder = keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=optimizer, loss="mae")

    # Training
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True, min_delta=0.00005
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=0.000001, verbose=1
    )
    history = autoencoder.fit(
        X_train_normal,
        X_train_normal,
        epochs=60,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        shuffle=True,
    )

    # Save model
    os.makedirs("./ml_models", exist_ok=True)
    autoencoder.save(model_path)
    print(f"Model saved to {model_path}")

# Calculate reconstruction error on training normal data for threshold
reconstructed_train = autoencoder.predict(X_train_normal, verbose=0)
mae_train_normal = np.mean(np.abs(X_train_normal - reconstructed_train), axis=1)
threshold = np.percentile(mae_train_normal, 85)
print(f"Using 95th percentile for threshold: {threshold:.4f}")
os.makedirs(thresholds_dir, exist_ok=True)
with open(threshold_file_path, "w") as f:
    f.write(str(threshold))

# Evaluate on test set
reconstructed_test = autoencoder.predict(X_test, verbose=0)
mae_test = np.mean(np.abs(X_test - reconstructed_test), axis=1)
pred_labels = (mae_test > threshold).astype(int)

# Metrics
from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_labels))
print("Classification Report:")
print(classification_report(y_test, pred_labels, digits=4))

# Detection rate, true positives, false positives
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
anomaly_df["reconstruction_error"] = mae_test[anomaly_indices]
os.makedirs("./dataset/anomalies", exist_ok=True)
anomaly_df.to_csv(
    "./dataset/anomalies/AE_anomalies_2024_12_25_test_augmented.csv", index=False
)
print(
    f"Saved {len(anomaly_df)} anomaly records to ./dataset/anomalies/AE_anomalies_2024_12_25_test_augmented.csv"
)
