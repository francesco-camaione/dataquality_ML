import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(f"Project root added to path: {project_root}")

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
from pyspark.sql.functions import (
    col,
    monotonically_increasing_id,
)
from pyspark.sql.types import StructType

# --- Scikit-learn Imports ---
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)

from app.utils.tf_utils import (
    configure_tensorflow,
    create_sequences_tf,
    calculate_percentile,
)
from app.utils.spark_utils import (
    create_spark_session,
)
from app.utils.data_utils import preprocess_data, process_batch_to_tensor
from app.utils.pipeline_utils import create_feature_pipeline
from app.utils.model_utils import create_model

configure_tensorflow()

table_name = "2024_12_bb_3d"
timesteps = 20
file_path = f"./dataset/{table_name}.csv"
spark_config = {
    "spark.driver.memory": "2g",
    "spark.executor.memory": "2g",
    "spark.memory.fraction": "0.8",
    "spark.memory.storageFraction": "0.3",
    "spark.sql.shuffle.partitions": "50",
    "spark.default.parallelism": "50",
    "spark.local.dir": "/tmp",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.memory.offHeap.enabled": "true",
    "spark.memory.offHeap.size": "1g",
    "spark.cleaner.periodicGC.interval": "10min",
    "spark.cleaner.referenceTracking.blocking": "true",
    "spark.cleaner.referenceTracking.cleanCheckpoints": "true",
    "spark.driver.maxResultSize": "1g",
    "spark.sql.broadcastTimeout": "300",
    "spark.network.timeout": "300s",
}
output_dir = "./dataset/anomalies"
os.makedirs(output_dir, exist_ok=True)
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# --- Spark Session ---
# Moved to create_spark_session in spark_utils.py
spark = create_spark_session(spark_config)


# --- Data Loading & Initial Processing ---
def load_data(spark_session, file_path):
    """Loads data from CSV and adds a record ID."""
    print(f"Loading data from: {file_path}")
    df = spark_session.read.option("header", "true").csv(file_path, inferSchema=True)
    df = df.withColumn("record_id", monotonically_increasing_id())
    print("Data loaded.")
    df.printSchema()
    return df


# --- Data Preparation Workflow Function ---
def prepare_sequences_for_model(
    normal_records_df,
    fraud_records_df,
    fitted_pipeline_model,
    timesteps_val,
):
    """Processes data, creates sequences, and splits into train/val/test sets."""
    print("--- Starting Data Preparation ---")
    print("Processing normal data batch...")
    normal_data_tensor, normal_record_ids = process_batch_to_tensor(
        normal_records_df, fitted_pipeline_model
    )
    if normal_data_tensor is None:
        raise ValueError("Failed to process normal data into tensor")
    print(f"Normal data tensor shape: {normal_data_tensor.shape}")

    print("Processing fraud data batch...")
    fraud_data_tensor, fraud_record_ids = process_batch_to_tensor(
        fraud_records_df, fitted_pipeline_model
    )
    # Handle case where fraud data processing might return None or empty tensor
    if fraud_data_tensor is None:
        # Attempt to get n_features from normal data tensor
        if normal_data_tensor is not None and len(normal_data_tensor.shape) > 1:
            n_features = normal_data_tensor.shape[1]
            print("Warning: No fraud data processed. Creating empty fraud tensor.")
            fraud_data_tensor = tf.zeros((0, n_features), dtype=tf.float32)
            fraud_record_ids = []
        else:
            raise ValueError(
                "Failed to process fraud data and could not determine feature count from normal data."
            )
    elif tf.shape(fraud_data_tensor)[0] == 0:
        print("Warning: Fraud data processed into an empty tensor.")
        fraud_record_ids = []  # Ensure IDs are empty if tensor is empty

    print(f"Fraud data tensor shape: {fraud_data_tensor.shape}")

    # Ensure normal_data_tensor is not empty before accessing shape
    if tf.shape(normal_data_tensor)[0] == 0:
        raise ValueError(
            "Normal data tensor is empty, cannot proceed with sequence creation or training."
        )

    n_features = normal_data_tensor.shape[1]
    print(f"Number of features: {n_features}")

    print("Creating sequences for normal data...")
    X_normal_sequences = create_sequences_tf(normal_data_tensor, timesteps_val)
    print(f"Normal sequences shape: {X_normal_sequences.shape}")

    # --- Train/Val/Test Split (Normal Data) ---
    num_normal_sequences = tf.shape(X_normal_sequences)[0]
    if num_normal_sequences == 0:
        raise ValueError("No normal sequences created, cannot split data.")

    train_fraction = 0.8
    val_fraction = 0.1  # Validation fraction of the *total* normal sequences
    # Test fraction is implicit: 1.0 - train_fraction - val_fraction

    train_size = tf.cast(
        tf.cast(num_normal_sequences, tf.float32) * train_fraction, tf.int32
    )
    val_size = tf.cast(
        tf.cast(num_normal_sequences, tf.float32) * val_fraction, tf.int32
    )

    # Ensure val_size is at least 1 if possible, and adjust train_size if needed
    val_size = tf.maximum(
        val_size,
        (
            tf.constant(1, dtype=tf.int32)
            if num_normal_sequences > train_size
            else tf.constant(0, dtype=tf.int32)
        ),
    )
    # Ensure train_size doesn't overlap with val_size due to rounding/minimums
    train_size = tf.minimum(train_size, num_normal_sequences - val_size)

    # Ensure train_size is at least 1 if possible
    train_size = tf.maximum(
        train_size,
        (
            tf.constant(1, dtype=tf.int32)
            if num_normal_sequences > val_size
            else tf.constant(0, dtype=tf.int32)
        ),
    )
    # Recalculate val_size in case train_size adjustment affected the boundary
    val_size = tf.minimum(val_size, num_normal_sequences - train_size)

    test_start_index = train_size + val_size

    X_train = X_normal_sequences[:train_size]
    X_val = X_normal_sequences[train_size:test_start_index]
    X_test_normal = X_normal_sequences[test_start_index:]

    print(f"Train sequences split: {tf.shape(X_train)[0]}")
    print(f"Validation sequences split: {tf.shape(X_val)[0]}")
    print(f"Test normal sequences split: {tf.shape(X_test_normal)[0]}")

    # --- Prepare Fraud Test Data ---
    print("Creating sequences for failure data...")
    X_test_fraud = tf.zeros(
        (0, timesteps_val, n_features), dtype=tf.float32
    )  # Default empty

    if tf.shape(fraud_data_tensor)[0] > 0:
        num_fraud_samples = tf.shape(fraud_data_tensor)[0]
        if num_fraud_samples < timesteps_val:
            print(
                f"Warning: Padding fraud data ({num_fraud_samples.numpy()} samples) with zeros for sequence creation (timesteps={timesteps_val})."
            )
            padding_size = timesteps_val - num_fraud_samples
            padding = tf.zeros((padding_size, n_features), dtype=tf.float32)
            fraud_data_tensor_padded = tf.concat([fraud_data_tensor, padding], axis=0)
            print(f"Padded fraud tensor shape: {fraud_data_tensor_padded.shape}")
            # Create sequences from padded data - should result in at least one sequence
            X_test_fraud = create_sequences_tf(fraud_data_tensor_padded, timesteps_val)
        else:
            # Enough samples, create sequences directly
            X_test_fraud = create_sequences_tf(fraud_data_tensor, timesteps_val)
    else:
        print("No fraud data samples to create sequences from.")

    print(f"Test fraud sequences shape: {X_test_fraud.shape}")
    print("--- Data Preparation Finished ---")

    return (
        X_train,
        X_val,
        X_test_normal,  # Keep for potential future use, though not used in current evaluation
        X_test_fraud,
        normal_record_ids,
        fraud_record_ids,
        n_features,
        normal_data_tensor,  # Pass back the original tensor for threshold calculation
    )


# --- Model Training Workflow Function ---
def train_lstm_ae_model(
    X_train_seq,
    X_val_seq,
    n_features_val,
    timesteps_val,
    model_name_prefix,  # Use a prefix instead of full table name directly
    strategy_val,
    epochs=10,
    batch_size=32,
    patience=3,
):
    """Creates, trains, and saves the LSTM Autoencoder model within a strategy scope."""
    print("--- Starting Model Training ---")
    if tf.shape(X_train_seq)[0] == 0 or tf.shape(X_val_seq)[0] == 0:
        raise ValueError("Training or validation data is empty. Cannot train model.")

    with strategy_val.scope():
        model = create_model(n_features_val, timesteps_val)

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train_seq, X_train_seq))
            .shuffle(1024)  # Adjust buffer size based on data
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_dataset = (
            tf.data.Dataset.from_tensor_slices((X_val_seq, X_val_seq))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        print(f"Training model for {epochs} epochs with patience {patience}...")
        checkpoint_path = f"./ml_models/{model_name_prefix}_checkpoint.keras"
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path, save_best_only=True, monitor="val_loss", verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=patience // 2 + 1,
                    min_lr=0.00001,
                    verbose=1,
                ),
                # Add TensorBoard callback for visualization
                # tf.keras.callbacks.TensorBoard(log_dir=f'./logs/{model_name_prefix}', histogram_freq=1)
            ],
            verbose=1,  # Or 2 for less output per epoch
        )

    # Load the best weights saved by ModelCheckpoint
    print(f"Loading best weights from checkpoint: {checkpoint_path}")
    try:
        model.load_weights(checkpoint_path)
    except Exception as e:
        print(
            f"Warning: Could not load weights from checkpoint {checkpoint_path}: {e}. Using weights from end of training."
        )

    model_save_path = f"./ml_models/{model_name_prefix}_final.keras"
    model.save(model_save_path)
    print(f"Final model saved to {model_save_path}")
    print("--- Model Training Finished ---")
    return model, history


# --- Prediction and Evaluation Workflow Function ---
def predict_and_evaluate(model, X_train_seq, X_test_normal, X_test_fr):
    """
    Predicts on test data (normal and fraud), calculates errors,
    and determines the threshold based on training errors.
    """
    print("--- Starting Evaluation ---")

    # --- Threshold Calculation (based on training data) ---
    print("Calculating threshold based on training reconstruction errors...")
    if tf.shape(X_train_seq)[0] == 0:
        raise ValueError(
            "Training data sequences are empty. Cannot calculate threshold."
        )

    try:
        train_pred = model.predict(X_train_seq)
        # Ensure inputs to subtraction are finite
        X_train_clean = tf.where(tf.math.is_nan(X_train_seq), 0.0, X_train_seq)
        train_pred_clean = tf.where(tf.math.is_nan(train_pred), 0.0, train_pred)

        normal_train_errors = tf.reduce_mean(
            tf.square(X_train_clean - train_pred_clean), axis=[1, 2]
        )
        # Check if errors calculation resulted in NaNs
        if tf.reduce_any(tf.math.is_nan(normal_train_errors)):
            print(
                "Warning: NaNs found in training reconstruction errors. Threshold calculation might be inaccurate."
            )
            normal_train_errors = tf.where(
                tf.math.is_nan(normal_train_errors),
                tf.zeros_like(normal_train_errors),
                normal_train_errors,
            )

        threshold_value_tensor = calculate_percentile(normal_train_errors, 95.0)
        threshold_value = threshold_value_tensor.numpy()
        print(
            f"Classification Threshold (95th percentile of train errors): {threshold_value:.4f}"
        )
    except Exception as e:
        print(f"Error calculating threshold: {e}")
        raise  # Re-raise the exception as threshold is critical

    # --- Prediction on Normal Test Data ---
    normal_test_errors = tf.constant([], dtype=tf.float32)  # Default empty
    if tf.shape(X_test_normal)[0] > 0:
        print(f"Predicting on {tf.shape(X_test_normal)[0]} normal test sequences...")
        try:
            X_test_normal_pred = model.predict(X_test_normal)
            X_test_normal_clean = tf.where(
                tf.math.is_nan(X_test_normal), 0.0, X_test_normal
            )
            X_test_normal_pred_clean = tf.where(
                tf.math.is_nan(X_test_normal_pred), 0.0, X_test_normal_pred
            )
            normal_test_errors = tf.reduce_mean(
                tf.square(X_test_normal_clean - X_test_normal_pred_clean), axis=[1, 2]
            )
            if tf.reduce_any(tf.math.is_nan(normal_test_errors)):
                print("Warning: NaNs found in normal test reconstruction errors.")
                normal_test_errors = tf.where(
                    tf.math.is_nan(normal_test_errors),
                    tf.zeros_like(normal_test_errors),
                    normal_test_errors,
                )
            print("Prediction on normal test data complete.")
        except Exception as e:
            print(f"Error predicting on normal test data: {e}")
            # Return empty tensor if prediction fails
            normal_test_errors = tf.constant([], dtype=tf.float32)
    else:
        print("No normal test sequences to predict on.")

    # --- Prediction on Fraud Test Data ---
    y_true_labels_fraud = tf.ones(
        tf.shape(X_test_fr)[0], dtype=tf.int32
    )  # All true labels are 1 (failure)
    fraud_test_errors = tf.constant([], dtype=tf.float32)  # Default empty

    if tf.shape(X_test_fr)[0] > 0:
        print(f"Predicting on {tf.shape(X_test_fr)[0]} failure test sequences...")
        try:
            X_test_fr_pred = model.predict(X_test_fr)
            X_test_fr_clean = tf.where(tf.math.is_nan(X_test_fr), 0.0, X_test_fr)
            X_test_fr_pred_clean = tf.where(
                tf.math.is_nan(X_test_fr_pred), 0.0, X_test_fr_pred
            )

            fraud_test_errors = tf.reduce_mean(
                tf.square(X_test_fr_clean - X_test_fr_pred_clean), axis=[1, 2]
            )
            if tf.reduce_any(tf.math.is_nan(fraud_test_errors)):
                print("Warning: NaNs found in fraud reconstruction errors.")
                fraud_test_errors = tf.where(
                    tf.math.is_nan(fraud_test_errors),
                    tf.zeros_like(fraud_test_errors),
                    fraud_test_errors,
                )

            print("Prediction and error calculation on failure data complete.")
        except Exception as e:
            print(f"Error predicting or calculating errors on fraud data: {e}")
            fraud_test_errors = tf.zeros_like(
                y_true_labels_fraud, dtype=tf.float32
            )  # Return zeros?
            print(
                "Proceeding with potentially zero reconstruction errors for fraud data."
            )
    else:
        print("No failure test sequences to predict on.")

    print("--- Evaluation Finished ---")
    # Return errors for both normal and fraud test sets
    return y_true_labels_fraud, fraud_test_errors, normal_test_errors, threshold_value


# --- Anomaly Mapping and Saving Workflow Function ---
def map_anomalies_and_save(
    reconstruction_errors_val,  # Tensor
    threshold,  # Float
    fraud_record_ids_all,  # List of IDs
    timesteps_val,  # Int
    spark_session,  # SparkSession
    original_df,  # DataFrame
    output_dir_val,  # String path
    output_filename_prefix,  # String prefix
):
    """Maps anomalous sequences (from failure data) back to original records and saves them."""
    print("--- Starting Anomaly Mapping and Saving ---")
    if not fraud_record_ids_all:
        print("No fraud record IDs provided, skipping anomaly mapping.")
        return

    if tf.shape(reconstruction_errors_val)[0] == 0:
        print(
            "No reconstruction errors provided for fraud data, skipping anomaly mapping."
        )
        return

    # Find indices of sequences where error exceeds threshold
    anomalous_sequence_indices_tensor = tf.where(reconstruction_errors_val > threshold)
    anomalous_sequence_indices = anomalous_sequence_indices_tensor.numpy().flatten()

    if anomalous_sequence_indices.size == 0:
        print("No anomalous sequences detected based on the threshold.")
        return

    print(
        f"Mapping {len(anomalous_sequence_indices)} anomalous sequences from failure data to original record IDs..."
    )

    anomalous_record_ids_set = set()
    record_id_to_error_map = {}  # Store max error for each record ID
    num_original_fraud_records = len(fraud_record_ids_all)

    # Iterate through the *indices* of the anomalous sequences
    for seq_idx in anomalous_sequence_indices:
        # The sequence starting at seq_idx corresponds to original records from
        # index seq_idx to seq_idx + timesteps_val - 1 in the *original* fraud data tensor/ID list.
        start_record_idx = seq_idx
        end_record_idx = seq_idx + timesteps_val  # Exclusive end

        # Clip the end index to the actual number of fraud records
        end_record_idx = min(end_record_idx, num_original_fraud_records)

        # Get the error for this specific sequence
        seq_error = reconstruction_errors_val[seq_idx].numpy()

        # Map the record indices within this range back to the original record IDs
        for i in range(start_record_idx, end_record_idx):
            record_id = fraud_record_ids_all[i]
            anomalous_record_ids_set.add(record_id)
            # Keep the highest reconstruction error associated with the record ID
            current_max_error = record_id_to_error_map.get(record_id, -1.0)
            record_id_to_error_map[record_id] = max(current_max_error, float(seq_error))

    num_unique_anomalous_records = len(anomalous_record_ids_set)
    print(
        f"Identified {num_unique_anomalous_records} unique failure records corresponding to {len(anomalous_sequence_indices)} anomalous sequences."
    )

    if anomalous_record_ids_set:
        # Convert the set to a list for Spark filtering
        anomalous_record_ids_list = list(anomalous_record_ids_set)

        # Filter the original DataFrame to get the full anomalous records
        # Ensure record_id column type matches for the filter
        anomalous_records_df = original_df.filter(
            col("record_id").isin(anomalous_record_ids_list)
        )

        # Create a Spark DataFrame for the errors to join
        error_map_pd = pd.DataFrame(
            list(record_id_to_error_map.items()),
            columns=["record_id", "max_reconstruction_error"],
        )
        error_map_spark = spark_session.createDataFrame(error_map_pd)

        # Join the original data with the max reconstruction error
        final_anomalies_df = anomalous_records_df.join(
            error_map_spark, "record_id", "left"
        ).orderBy(col("max_reconstruction_error").desc())

        print("Detected Anomalies Sample (Original Columns + Max Error):")
        final_anomalies_df.show(truncate=False)

        # Save the results
        anomaly_csv_path = os.path.join(
            output_dir_val, f"{output_filename_prefix}_anomalies.csv"
        )
        try:
            final_anomalies_df.coalesce(1).write.csv(
                anomaly_csv_path, header=True, mode="overwrite"
            )
            print(f"Detected anomalies saved to: {anomaly_csv_path}")
        except Exception as e:
            print(f"Error saving anomalies CSV: {e}")
    else:
        # This case is already handled by the check at the start of the block
        pass

    print("--- Anomaly Mapping and Saving Finished ---")


# --- Metrics Calculation and Plotting Workflow Function ---
def calculate_and_plot_metrics(
    y_true_labels_fraud,  # Tensor (all 1s for fraud data)
    fraud_scores_val,  # Tensor (reconstruction errors for failures)
    normal_scores_val,  # Tensor (reconstruction errors for normal test data)
    threshold_val,  # Float (pre-calculated threshold)
    total_actual_failures,  # Int (count from raw fraud data)
    plot_dir_val,  # String path
    plot_filename_prefix,  # String prefix
):
    """Calculates detection metrics, ROC curve, and plots ROC."""
    print("--- Starting Metrics Calculation and Plotting ---")

    fraud_scores = fraud_scores_val.numpy()
    normal_scores = normal_scores_val.numpy()

    # --- Failure Detection Metrics (based on pre-calculated threshold) ---
    detected_failures = 0
    if fraud_scores.size > 0:
        y_pred_labels_fraud_at_threshold = (fraud_scores > threshold_val).astype(int)
        detected_failures = np.sum(y_pred_labels_fraud_at_threshold)
    else:
        print("No fraud prediction scores available for threshold-based metrics.")

    missed_failures = total_actual_failures - detected_failures
    missed_failures = max(0, missed_failures)

    print(f"Metrics based on pre-calculated threshold ({threshold_val:.4f}):")
    print(f"  Total Failure Records Expected (from input): {total_actual_failures}")
    print(f"  Failure Test Sequences Evaluated: {fraud_scores.size}")
    print(f"  Failures Detected as Anomalies (TP at threshold): {detected_failures}")
    print(f"  Failures Missed (FN at threshold): {missed_failures}")

    if total_actual_failures > 0:
        detection_rate_at_threshold = (detected_failures / total_actual_failures) * 100
        print(
            f"  Failure Detection Rate (at threshold): {detection_rate_at_threshold:.2f}%"
        )
    else:
        print(
            "  No actual failure records for detection rate calculation at threshold."
        )

    # --- ROC Curve Calculation and Plotting ---
    if normal_scores.size > 0 and fraud_scores.size > 0:
        print("\nCalculating ROC curve...")
        # True labels: 0 for normal, 1 for fraud
        y_true_roc = np.concatenate(
            [np.zeros(len(normal_scores)), np.ones(len(fraud_scores))]
        )
        # Scores: reconstruction errors
        y_scores_roc = np.concatenate([normal_scores, fraud_scores])

        fpr, tpr, thresholds_roc = roc_curve(y_true_roc, y_scores_roc)
        roc_auc_value = auc(fpr, tpr)

        print(f"Area Under ROC Curve (AUC): {roc_auc_value:.4f}")

        plt.figure(figsize=(10, 7))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc_value:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic (ROC) - {plot_filename_prefix}")
        plt.legend(loc="lower right")
        plt.grid(True)
        roc_plot_path = os.path.join(
            plot_dir_val, f"{plot_filename_prefix}_ROC_curve.png"
        )
        try:
            plt.savefig(roc_plot_path)
            print(f"ROC curve plot saved to: {roc_plot_path}")
        except Exception as e:
            print(f"Error saving ROC plot: {e}")
        finally:
            plt.close()

        # --- Confusion Matrix (at the pre-calculated threshold) ---
        # For the CM, we need predictions on normal data using the same threshold
        if normal_scores.size > 0:
            predictions_on_normal_at_threshold = (normal_scores > threshold_val).astype(
                int
            )
            all_true_labels_cm = y_true_roc  # Same as for ROC
            all_predictions_cm = np.concatenate(
                [predictions_on_normal_at_threshold, y_pred_labels_fraud_at_threshold]
            )

            print(f"\nConfusion Matrix (at threshold {threshold_val:.4f}):")
            try:
                cm = confusion_matrix(all_true_labels_cm, all_predictions_cm)
                print("         Predicted Normal  Predicted Failure")
                print(f"Actual Normal    {cm[0,0]:<15}  {cm[0,1]:<17}")  # TN, FP
                print(f"Actual Failure   {cm[1,0]:<15}  {cm[1,1]:<17}")  # FN, TP
            except Exception as e:
                print(f"Could not generate confusion matrix: {e}")
        else:
            print("\nSkipping confusion matrix: Normal scores not available.")

    elif normal_scores.size == 0:
        print("\nSkipping ROC curve calculation: Normal test scores are not available.")
    elif fraud_scores.size == 0:
        print("\nSkipping ROC curve calculation: Fraud test scores are not available.")

    print("--- Metrics Calculation and Plotting Finished ---")


def main():
    """Main execution function."""
    print("Starting LSTM Autoencoder for Anomaly Detection...")

    # --- Load Data ---
    df_all = load_data(spark, file_path)
    raw_total_count = df_all.count()
    print(f"Total records loaded: {raw_total_count}")

    normal_records_raw = df_all.where(col("failure") == 0)
    fraud_records_raw = df_all.where(col("failure") == 1)
    normal_raw_count = normal_records_raw.count()
    fraud_raw_count = fraud_records_raw.count()
    print(f"Raw normal records count (limited): {normal_raw_count}")
    print(f"Raw failure records count: {fraud_raw_count}")

    if normal_raw_count == 0:
        print(
            "Error: No normal records found after filtering and limiting. Cannot train model."
        )
        return  # Exit if no training data

    # --- Preprocess Data --- #
    print("Preprocessing normal data...")
    # Preprocess normal data first to establish the reference schema
    normal_records = preprocess_data(normal_records_raw)
    reference_schema = normal_records.schema
    print(f"Reference schema established with {len(reference_schema.fields)} fields.")
    normal_records.cache()  # Cache preprocessed normal data
    normal_processed_count = (
        normal_records.count()
    )  # Count after potential null column removal
    print(f"Processed normal records count: {normal_processed_count}")

    if normal_processed_count == 0:
        print(
            "Error: No normal records remaining after preprocessing. Cannot train model."
        )
        return

    # Preprocess fraud data, aligning to the reference schema
    print("Preprocessing failure data...")
    fraud_records = preprocess_data(fraud_records_raw, reference_schema)
    fraud_records.cache()  # Cache preprocessed fraud data
    fraud_processed_count = fraud_records.count()
    print(f"Processed failure records count: {fraud_processed_count}")

    # --- Feature Engineering --- #
    print("Creating and fitting feature pipeline...")
    # Create pipeline based on the schema of *processed* normal data
    pipeline = create_feature_pipeline(normal_records.schema)
    fitted_pipeline = pipeline.fit(normal_records)
    print("Feature pipeline fitted.")
    # Optional: Save the fitted pipeline
    # fitted_pipeline.save(f"./ml_models/feature_pipeline_{table_name}")

    # Unpersist raw DFs if no longer needed
    normal_records_raw.unpersist()
    fraud_records_raw.unpersist()

    # --- Prepare Data for TF Model --- #
    (
        X_train,
        X_val,
        X_test_normal,  # Kept but unused in current evaluation
        X_test_fraud,
        normal_record_ids,
        fraud_record_ids,
        n_features,
        normal_data_tensor,  # Needed for threshold calc
    ) = prepare_sequences_for_model(
        normal_records,  # Use processed data
        fraud_records,  # Use processed data
        fitted_pipeline,
        timesteps,
    )

    # Unpersist processed DFs after tensor conversion
    normal_records.unpersist()
    fraud_records.unpersist()
    gc.collect()  # Suggest garbage collection

    # --- Model Training --- #
    strategy = tf.distribute.MirroredStrategy()
    print(f"Training using strategy: {strategy}")
    model, history = train_lstm_ae_model(
        X_train,
        X_val,
        n_features,
        timesteps,
        f"LSTM_AE_{table_name}",  # Pass prefix for model naming
        strategy,
        epochs=10,  # Make epochs configurable
        patience=3,  # Make patience configurable
    )

    # --- Evaluation --- #
    y_true_labels_fraud, fraud_scores_val, normal_scores_val, threshold = (
        predict_and_evaluate(
            model, X_train, X_test_normal, X_test_fraud  # Pass all required data
        )
    )

    # --- Anomaly Mapping --- #
    map_anomalies_and_save(
        fraud_scores_val,  # Use fraud scores for anomaly mapping
        threshold,
        fraud_record_ids,
        timesteps,
        spark,
        df_all,
        output_dir,
        f"LSTM_AE_{table_name}",
    )

    # --- Metrics Reporting --- #
    calculate_and_plot_metrics(
        y_true_labels_fraud,  # Labels corresponding to fraud_scores_val
        fraud_scores_val,  # Errors for fraud data
        normal_scores_val,  # Errors for normal data
        threshold,
        fraud_raw_count,
        plot_dir,
        f"LSTM_AE_{table_name}",
    )

    print("LSTM Autoencoder script finished.")


# --- Script Entry Point --- #
if __name__ == "__main__":
    main()
