import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
import keras
import numpy as np
from lib.utils import (
    ms_error_ae,
    infer_column_types_from_schema,
)

table_name = "fraud_detection_sample"

spark = (
    SparkSession.builder.appName("App")
    .master("local[*]")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.executor.memory", "8g")
    .getOrCreate()
)

# Load and split dataset
raw_df = spark.read.option("header", "true").csv(
    f"./dataset/{table_name}.csv", inferSchema=True
)

# Separate normal vs fraud
df_norm = raw_df.where(raw_df["isFraud"] == 0)
df_fraud = raw_df.where(raw_df["isFraud"] == 1)

# For anomaly‐table output we'll need raw Pandas too
raw_norm_pdf = df_norm.toPandas()
raw_fraud_pdf = df_fraud.toPandas()

# Infer schema & build feature pipeline
categorical_cols, numerical_cols = infer_column_types_from_schema(df_norm.schema)

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

fitted_pipeline = pipeline.fit(df_norm)
proc_norm_df = fitted_pipeline.transform(df_norm)

fitted_pipeline2 = pipeline.fit(df_fraud)
proc_fraud_df = fitted_pipeline2.transform(df_fraud)

# Convert to numpy arrays
norm_pdf = proc_norm_df.select("features").toPandas()
fraud_pdf = proc_fraud_df.select("features").toPandas()

# Collect each row's Spark Vector (from the "features" column), convert to a NumPy array,
# and vertically stack them into a single 2D float32 array
X_train = np.vstack(norm_pdf["features"].apply(lambda v: v.toArray()).tolist()).astype(
    np.float32
)
X_test = np.vstack(fraud_pdf["features"].apply(lambda v: v.toArray()).tolist()).astype(
    np.float32
)


# train the model
# input_dim = X_train.shape[1]
# input_layer = keras.layers.Input(shape=(input_dim,))
#
# # Encoder
# encoded = keras.layers.Dense(128, activation="relu")(input_layer)
# encoded = keras.layers.BatchNormalization()(encoded)
# encoded = keras.layers.Dropout(0.2)(encoded)
# encoded = keras.layers.Dense(64, activation="relu")(encoded)
# encoded = keras.layers.BatchNormalization()(encoded)
# encoded = keras.layers.Dropout(0.2)(encoded)
# latent_space = keras.layers.Dense(32, activation="relu")(encoded)
# latent_space = keras.layers.BatchNormalization()(latent_space)
#
# # Decoder
# decoded = keras.layers.Dense(64, activation="relu")(latent_space)
# decoded = keras.layers.BatchNormalization()(decoded)
# decoded = keras.layers.Dropout(0.2)(decoded)
# decoded = keras.layers.Dense(128, activation="relu")(decoded)
# decoded = keras.layers.BatchNormalization()(decoded)
# decoded = keras.layers.Dropout(0.2)(decoded)
# decoded = keras.layers.Dense(input_dim, activation="sigmoid")(decoded)
#
# autoencoder = keras.models.Model(inputs=input_layer, outputs=decoded)
#
# initial_lr = 0.0001
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_lr, decay_steps=1000, decay_rate=0.9
# )
# optimizer = keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
# autoencoder.compile(optimizer=optimizer, loss="mean_squared_error")
#
# early_stopping = keras.callbacks.EarlyStopping(
#     monitor="val_loss", patience=10, restore_best_weights=True
# )
#
# history = autoencoder.fit(
#     X_train, X_train,
#     epochs=100, batch_size=128,
#     validation_split=0.2,
#     callbacks=[early_stopping],
# )
#
# autoencoder.save(f"./ml_models/AE_{table_name}.keras")

#  Load trained model & predict
autoencoder = keras.models.load_model(f"./ml_models/AE_{table_name}.keras")
reconstructed = autoencoder.predict(X_test)
mse = ms_error_ae(X_test, reconstructed)

reconstructed_train = autoencoder.predict(X_train)
mse_train_normal = ms_error_ae(X_train, reconstructed_train)

# Threshold & Anomaly Detection - using normal data distribution
threshold = np.percentile(mse_train_normal, 95)  # Using 96th percentile of normal data
anomalies = mse > threshold

# Metrics
detected = np.sum(anomalies)
total = len(X_test)
rate_pct = 100 * detected / total

print(f"Anomaly threshold: {threshold:.4f}")
print(f"Test fraud records: {total}")
print(f"Fraud correctly flagged: {detected}")
print(f"Detection rate: {rate_pct:.2f}%")

# Build & print anomaly table
anomaly_idxs = np.where(anomalies)[0]
anomaly_table = raw_fraud_pdf.iloc[anomaly_idxs]
# Add reconstruction error to the anomaly table
anomaly_table["reconstruction_error"] = mse[anomaly_idxs]
# anomaly_table = anomaly_table.sort_values(by='reconstruction_error', ascending=False)

print("Anomaly table (raw fraud records flagged):")
print(anomaly_table)

# Model statistics and ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# 1. Get reconstruction errors for the training (normal) data
reconstructed_train = autoencoder.predict(X_train)
mse_train_normal = ms_error_ae(X_train, reconstructed_train)  # MSE for normal data

# mse is already calculated for X_test (fraud data) in the existing code
mse_test_fraud = mse

# 2. Create combined true labels and scores for ROC analysis
# True labels: 0 for normal (X_train), 1 for fraud (X_test)
y_true = np.concatenate([np.zeros(len(mse_train_normal)), np.ones(len(mse_test_fraud))])

# Scores: reconstruction errors
y_scores = np.concatenate([mse_train_normal, mse_test_fraud])

# 3. Calculate ROC curve and AUC
fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

print(f"Area Under ROC Curve (AUC): {roc_auc:.4f}")

# 4. Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Receiver Operating Characteristic (ROC) - {table_name}")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(f"./plots/ROC_Curve_AE_{table_name}.png")
print(f"ROC curve plot saved to: /plots/ROC_Curve_AE_{table_name}.png")
# plt.show() # Uncomment if you want to display the plot

# 5. Calculate and print Confusion Matrix and Classification Report
# Using the threshold based on normal data distribution
# Predictions for normal data using this threshold
predictions_on_normal_data = mse_train_normal > threshold
# Predictions on fraud data
# anomalies = mse > threshold (already calculated above)

all_predictions = np.concatenate([predictions_on_normal_data, anomalies])

print("\\nConfusion Matrix (Threshold: {:.4f}):".format(threshold))
# Rows: True Normal, True Fraud
# Cols: Predicted Normal, Predicted Fraud
# Correct order for confusion_matrix: TN, FP, FN, TP
# Our labels: 0=Normal, 1=Fraud.
# Our predictions: False=Predicted Normal, True=Predicted Fraud
# So, if y_true=0 (Normal) and y_pred=False (Predicted Normal) -> TN
# if y_true=0 (Normal) and y_pred=True (Predicted Fraud) -> FP
# if y_true=1 (Fraud) and y_pred=False (Predicted Normal) -> FN
# if y_true=1 (Fraud) and y_pred=True (Predicted Fraud) -> TP
cm = confusion_matrix(y_true, all_predictions)
print(cm)

spark.stop()
