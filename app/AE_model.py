import sys
import os

from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from pyspark.sql import functions, types
import keras
import numpy as np
from lib.connector import SparkToAWS
from lib.utils import (
    mae_error_ae,
    boolean_columns,
    build_and_fit_feature_pipeline,
)

table_name = "2024_12_bb_3d"

connector = SparkToAWS()
spark = connector.create_local_spark_session()

raw_df = (
    spark.read.option("header", "true").csv(
        f"./dataset/{table_name}.csv", inferSchema=True
    )
).repartition(32)

print("df has ", raw_df.count(), " records")

raw_boolean_cols = boolean_columns(raw_df.schema)

if raw_boolean_cols:
    print(f"Casting boolean columns to integer: {raw_boolean_cols}")
    for column in raw_boolean_cols:
        raw_df = raw_df.withColumn(
            column, functions.col(column).cast(types.IntegerType())
        )

df_norm = raw_df.where(raw_df["failure"] == 0)
df_fraud = raw_df.where(raw_df["failure"] == 1)
print("number of failure records in the raw_df: ", df_fraud.count())

fitted_pipeline, feature_cols_from_util = build_and_fit_feature_pipeline(
    df_norm, raw_boolean_cols
)

pipeline_path = f"./pipelines/AE_pipeline/pipeline_{table_name}"
try:
    print(f"\nSaving fitted pipeline to {pipeline_path}...")
    os.makedirs("./pipelines/AE_pipeline", exist_ok=True)
    fitted_pipeline.write().overwrite().save(pipeline_path)
    print(f"Saved pipeline to {pipeline_path}")
except Exception as e:
    print(f"Error saving pipeline: {str(e)}")


proc_norm_df = fitted_pipeline.transform(df_norm)

proc_fraud_df = fitted_pipeline.transform(df_fraud)

norm_vectors = []
for row in proc_norm_df.select("features").collect():
    norm_vectors.append(row.features.toArray())
X_train = np.vstack(norm_vectors).astype(np.float32)

fraud_vectors = []
if df_fraud.count() > 0 and proc_fraud_df.count() > 0:
    for row in proc_fraud_df.select("features").collect():
        fraud_vectors.append(row.features.toArray())
    if fraud_vectors:
        X_test = np.vstack(fraud_vectors).astype(np.float32)
    else:
        print(
            "Warning: No fraud vectors collected after processing, X_test will be empty."
        )
        # Create an empty array with the correct number of features if X_train is available
        # This helps avoid errors later if X_test is expected to have a certain shape for predictions
        num_features = X_train.shape[1] if X_train.size > 0 else 0
        X_test = np.empty((0, num_features), dtype=np.float32)
else:
    print(
        "No fraud records to process or fraud DataFrame became empty after processing."
    )
    num_features = X_train.shape[1] if X_train.size > 0 else 0
    X_test = np.empty((0, num_features), dtype=np.float32)


raw_df.unpersist()  # Unpersist raw_df as it's no longer needed directly

model_path = f"./ml_models/AE_{table_name}.keras"
thresholds_dir = "./dataset/thresholds"
threshold_file_path = os.path.join(thresholds_dir, f"AE_threshold_{table_name}.txt")
os.makedirs(thresholds_dir, exist_ok=True)  # Ensure threshold directory exists
model_was_loaded = os.path.exists(model_path)

if not model_was_loaded:
    if X_train.shape[0] == 0:
        print("Error: X_train is empty. Cannot train the model.")
        spark.stop()
        sys.exit(1)
    if X_train.shape[1] == 0:
        print(
            f"Error: X_train has 0 features (shape: {X_train.shape}). Check pipeline and feature_cols: {feature_cols_from_util}"
        )
        spark.stop()
        sys.exit(1)

    print(
        f"\nTraining new Keras autoencoder model. Input dimension: {X_train.shape[1]}"
    )
    # train the model
    input_dim = X_train.shape[1]
    input_layer = keras.layers.Input(shape=(input_dim,))

    # Encoder with L2 regularization and noise for better generalization
    input_with_noise = keras.layers.GaussianNoise(0.01)(input_layer)

    # Encoder path with gradually decreasing dimensions
    encoded = keras.layers.Dense(
        512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.00025)
    )(input_with_noise)
    encoded = keras.layers.BatchNormalization()(encoded)

    encoded = keras.layers.Dense(
        256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.00025)
    )(encoded)
    encoded = keras.layers.BatchNormalization()(encoded)

    encoded = keras.layers.Dense(
        128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.00025)
    )(encoded)
    encoded = keras.layers.BatchNormalization()(encoded)

    # Deeper compression for better feature extraction
    latent_space = keras.layers.Dense(
        48, activation="relu", kernel_regularizer=keras.regularizers.l2(0.00025)
    )(encoded)
    latent_space = keras.layers.BatchNormalization()(latent_space)

    # Decoder path with gradually increasing dimensions
    decoded = keras.layers.Dense(
        128, activation="linear", kernel_regularizer=keras.regularizers.l2(0.00025)
    )(latent_space)
    decoded = keras.layers.BatchNormalization()(decoded)

    decoded = keras.layers.Dense(
        256, activation="linear", kernel_regularizer=keras.regularizers.l2(0.00025)
    )(decoded)
    decoded = keras.layers.BatchNormalization()(decoded)

    decoded = keras.layers.Dense(
        512, activation="linear", kernel_regularizer=keras.regularizers.l2(0.00025)
    )(decoded)
    decoded = keras.layers.BatchNormalization()(decoded)

    # Final reconstruction
    decoded = keras.layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = keras.models.Model(inputs=input_layer, outputs=decoded)

    initial_lr = 0.0001
    optimizer = keras.optimizers.legacy.Adam(learning_rate=initial_lr)
    autoencoder.compile(optimizer=optimizer, loss="mae")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        min_delta=0.00005,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=0.000001,
        verbose=1,
    )

    history = autoencoder.fit(
        X_train,
        X_train,
        epochs=60,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        shuffle=True,
    )

    print("\nCalculating reconstruction error distribution...")
    reconstructed_train = autoencoder.predict(X_train, verbose=0)
    mae_train_normal = mae_error_ae(X_train, reconstructed_train)

    threshold = np.percentile(mae_train_normal, 95)
    print(f"Using 95th percentile for threshold: {threshold:.4f}")

    # Print additional statistics about reconstruction errors
    print(f"\nReconstruction Error Statistics:")
    print(f"Mean: {np.mean(mae_train_normal):.4f}")
    print(f"Median: {np.median(mae_train_normal):.4f}")
    print(f"Std Dev: {np.std(mae_train_normal):.4f}")
    print(f"Min: {np.min(mae_train_normal):.4f}")
    print(f"Max: {np.max(mae_train_normal):.4f}")

    autoencoder.save(f"./ml_models/AE_{table_name}.keras")
    print(f"Keras model saved to {model_path}")

print(f"\nLoading Keras model from {model_path}...")
autoencoder = keras.models.load_model(model_path, compile=False)
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)

autoencoder.compile(optimizer=optimizer, loss="mae")

if X_train.shape[0] > 0:
    reconstructed_train = autoencoder.predict(X_train, verbose=0)
    mae_train_normal = mae_error_ae(X_train, reconstructed_train)
else:
    print("X_train is empty, cannot calculate threshold.")
    mae_train_normal = np.array([])

print("Predicting on test (failure) data...")
if X_test.shape[0] > 0:
    reconstructed_test = autoencoder.predict(X_test, verbose=0)
    mae_test_fraud = mae_error_ae(X_test, reconstructed_test)
else:
    print("X_test is empty. No failure data to predict on.")
    mae_test_fraud = np.array([])  # Keep as mae_test_fraud for clarity below


if not model_was_loaded and mae_train_normal.size > 0:
    threshold = np.percentile(mae_train_normal, 95)
    with open(threshold_file_path, "w") as f:
        f.write(str(threshold))
    print(
        f"Threshold calculated for new model ({threshold:.4f}) and saved to: {threshold_file_path}"
    )
elif mae_train_normal.size == 0 and not model_was_loaded:
    print("Cannot calculate new threshold as mae_train_normal is empty.")
    threshold = 0  # Or handle error appropriately
elif os.path.exists(threshold_file_path):
    with open(threshold_file_path, "r") as f:
        threshold = float(f.read().strip())
        print(f"Threshold loaded from {threshold_file_path}: {threshold:.4f}")
else:
    print(
        f"Threshold file {threshold_file_path} not found and model was not re-trained. Using default threshold or error."
    )
    threshold = 0  # Or handle error: perhaps try to calculate if mae_train_normal exists but model_was_loaded is true

if X_test.shape[0] > 0:
    anomalies = mae_test_fraud > threshold
    detected = np.sum(anomalies)
    total_fraud_records = X_test.shape[0]
    detection_rate_pct = (
        (100 * detected / total_fraud_records) if total_fraud_records > 0 else 0
    )
else:
    detected = 0
    total_fraud_records = 0
    detection_rate_pct = 0

print(f"\nAnomaly threshold: {threshold:.4f}")
print(f"Test fraud records processed: {total_fraud_records}")
print(f"Fraud correctly flagged by model: {detected}")
print(f"Detection rate on fraud data: {detection_rate_pct:.2f}%")

# Build & print anomaly table
if total_fraud_records > 0 and detected > 0:
    anomaly_idxs = np.where(anomalies)[0]
    # Handle timestamp columns before converting to Pandas
    df_fraud_for_pandas = df_fraud  # df_fraud should still be in scope
    for col_name, dtype_val in df_fraud_for_pandas.dtypes:
        if (
            dtype_val == "timestamp" or dtype_val == "timestamp_ntz"
        ):  # Handling both common timestamp types
            df_fraud_for_pandas = df_fraud_for_pandas.withColumn(
                col_name, functions.col(col_name).cast("string")
            )
    raw_fraud_pdf = df_fraud_for_pandas.toPandas().iloc[anomaly_idxs]
 
    raw_fraud_pdf["reconstruction_error"] = mae_test_fraud[anomaly_idxs]

    print("\nAnomaly table (fraud records flagged by model):")
    print(raw_fraud_pdf.head())
    os.makedirs("./dataset/anomalies", exist_ok=True)
    raw_fraud_pdf.to_csv(f"./dataset/anomalies/anomalies_{table_name}.csv", index=False)
    print(f"Anomaly table saved to ./dataset/anomalies/anomalies_{table_name}.csv")
else:
    print("\nNo fraud anomalies detected or no fraud data to analyze.")

if mae_train_normal.size > 0 and mae_test_fraud.size > 0:
    y_true = np.concatenate(
        [np.zeros(len(mae_train_normal)), np.ones(len(mae_test_fraud))]
    )
    y_scores = np.concatenate([mae_train_normal, mae_test_fraud])

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"Area Under ROC Curve (AUC): {roc_auc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) - {table_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/ROC_Curve_AE_{table_name}.png")
    print(f"ROC curve plot saved to: ./plots/ROC_Curve_AE_{table_name}.png")

connector.close_spark_session()
