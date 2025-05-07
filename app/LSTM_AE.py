import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from keras.models import Model, load_model
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from keras.optimizers.legacy import Adam
from lib.utils import create_sequences, ms_error, infer_column_types_from_schema
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    confusion_matrix,
)
import pandas as pd

table_name = "fraud_detection_sample"
timesteps = 20

# Initialize Spark session
spark = (
    SparkSession.builder.appName("App")
    .master("local[*]")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.executor.memory", "8g")
    .getOrCreate()
)

# Load full dataset without initial limit
full_original_df = spark.read.option("header", "true").csv(
    f"./dataset/{table_name}.csv", inferSchema=True
)

# Split data: normal for training, fraud for testing
fraud_records = full_original_df.where(full_original_df["isFraud"] == 1)
normal_records = full_original_df.where(full_original_df["isFraud"] == 0).limit(20000)

# Prepare feature pipeline
categorical_cols, numerical_cols = infer_column_types_from_schema(normal_records.schema)
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="skip")
    for c in categorical_cols
]
feature_cols = [c + "_idx" for c in categorical_cols] + numerical_cols

assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")
scaler = StandardScaler(
    inputCol="assembled_features", outputCol="features", withMean=True, withStd=True
)

pipeline = Pipeline(stages=indexers + [assembler, scaler])
fitted_pipeline = pipeline.fit(normal_records)

# Process normal data
processed_df = fitted_pipeline.transform(normal_records)
pdf = processed_df.select("features").toPandas()
feature_list = pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
data = np.array(feature_list, dtype=np.float32)

# Create sequences and split into train/test
X = create_sequences(data, timesteps)
train_size = int(0.8 * len(X))
X_train = X[:train_size]
X_test_normal = X[train_size:]

# Process fraud data for testing
fitted_fraud_pipeline = pipeline.fit(fraud_records)
processed_fraud_df = fitted_fraud_pipeline.transform(fraud_records)
fraud_pdf = processed_fraud_df.select("features").toPandas()

fraud_features = fraud_pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
fraud_data = np.array(fraud_features, dtype=np.float32)
X_test_fraud = create_sequences(fraud_data, timesteps)

# Combine normal and fraud test data with labels
X_test_combined = np.concatenate([X_test_normal, X_test_fraud], axis=0)
y_test_combined = np.concatenate(
    [
        np.zeros(X_test_normal.shape[0]),  # 0 for normal
        np.ones(X_test_fraud.shape[0]),  # 1 for fraud
    ]
).astype(int)


# # Define and train model
# n_features = X_train.shape[2]
# input_seq = Input(shape=(timesteps, n_features))

# # Encoder
# encoded = LSTM(32, activation="relu", return_sequences=True)(input_seq)
# encoded = LSTM(16, activation="relu")(encoded)

# # Latent space
# decoded = RepeatVector(timesteps)(encoded)

# # Decoder
# decoded = LSTM(16, activation="relu", return_sequences=True)(decoded)
# decoded = LSTM(32, activation="relu", return_sequences=True)(decoded)
# decoded = TimeDistributed(Dense(n_features))(decoded)

# # Build and compile model
# lstm_ae = Model(inputs=input_seq, outputs=decoded)
# lstm_ae.compile(optimizer=Adam(learning_rate=0.005), loss="mse")

# # Train
# history = lstm_ae.fit(
#     X_train,
#     X_train,
#     epochs=11,
#     batch_size=32,
#     validation_split=0.2,
#     shuffle=True,
# )

# lstm_ae.save(f"./ml_models/LSTM_AE_{table_name}.keras")
# model = lstm_ae

model = load_model(f"./ml_models/LSTM_AE_{table_name}.keras")

# Make predictions on test data
X_test_pred = model.predict(X_test_combined)
reconstruction_errors = ms_error(X_test_combined, X_test_pred)

# Make predictions on normal data only
X_train_pred = model.predict(X_train)
normal_reconstruction_errors = ms_error(X_train, X_train_pred)

# Calculate threshold based on normal data reconstruction errors
threshold = np.percentile(normal_reconstruction_errors, 95)
print(f"Classification Threshold (95th percentile of normal data): {threshold:.4f}")

# Apply threshold to identify anomalies
anomalies = reconstruction_errors > threshold

# For a real-world scenario, we need to track which original records correspond to our test sequences
# Create a DataFrame with all records and their reconstruction errors
# Convert full dataset to pandas for easier handling
original_df = fraud_records.toPandas()

# Create empty list to store detected anomaly records
detected_anomalies = []

for i in range(len(X_test_combined)):
    if anomalies[i]:  # Only check if it's an anomaly, not if it's a known fraud
        record_idx = i % len(original_df)
        record = original_df.iloc[record_idx].to_dict()

        # Add the reconstruction error
        record["error"] = reconstruction_errors[i]
        detected_anomalies.append(record)

# Create and display anomaly table
anomaly_table = pd.DataFrame(detected_anomalies)
print(f"\nDetected Anomalies ({len(detected_anomalies)}):")
print(anomaly_table)

# Calculate evaluation metrics
if len(np.unique(y_test_combined)) > 1:
    # ROC AUC
    auc_score = roc_auc_score(y_test_combined, reconstruction_errors)
    print(f"AUC ROC Score: {auc_score:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_combined, reconstruction_errors)
    plt.figure(figsize=(10, 7))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (area = {auc_score:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - LSTM AE on {table_name}")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.savefig(f"./plots/ROC_Curve_LSTM_AE_{table_name}.png")
    print(f"ROC curve saved to /plots/ROC_Curve_LSTM_AE_{table_name}.png")

    # Confusion Matrix
    predicted_labels = anomalies.astype(int)
    cm = confusion_matrix(y_test_combined, predicted_labels, labels=[0, 1])
    print("Confusion Matrix (Labels: 0=Normal, 1=Fraud):")
    print(cm)

    # Classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_combined,
        predicted_labels,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    print(f"Precision (Fraud): {precision:.4f}")
    print(f"Recall (Fraud): {recall:.4f}")
    print(f"F1-Score (Fraud): {f1:.4f}")


spark.stop()
