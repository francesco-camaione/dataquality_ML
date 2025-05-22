import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import tensorflow as tf
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer
from keras.models import load_model
from pyspark.sql.functions import col, count, when, stddev_pop, udf
from pyspark.sql.types import NumericType, BooleanType
from lib.utils import (
    create_sequences,
    mae_error,
    infer_column_types_from_schema,
    plot_roc_curve,
    boolean_columns,
)


def test_lstm_ae_model(test_table_name, train_table_name="2024_12_bb_3d", timesteps=20):
    """
    Test the LSTM AE model on a dataset containing both failure and non-failure records.
    """
    # Initialize Spark session
    spark = (
        SparkSession.builder.appName("TestLSTMAE")
        .master("local[*]")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    # Load the test dataset
    print(f"Loading test dataset: {test_table_name}")
    df = spark.read.option("header", "true").csv(
        f"./dataset/{test_table_name}.csv",
        inferSchema=True,
    )
    print(f"Initial DataFrame row count: {df.count()}")
    print(f"Initial failure count in df: {df.where(col('failure') == 1).count()}")

    # Drop columns that are entirely NULL
    cols_to_drop = []
    for c_name in df.columns:
        non_null_count = df.where(col(c_name).isNotNull()).count()
        if non_null_count == 0:
            cols_to_drop.append(c_name)
            print(f"Column '{c_name}' contains only NULL values and will be dropped.")

    if cols_to_drop:
        df = df.drop(*cols_to_drop)
        print(f"Dropped columns with all NULLs: {cols_to_drop}")

    # Convert boolean columns to numeric type
    boolean_cols = boolean_columns(df.schema)
    if boolean_cols:
        print(f"Converting boolean columns to numeric: {boolean_cols}")
        for column in boolean_cols:
            df = df.withColumn(column, col(column).cast("integer"))

    # Split data into normal and failure
    df_normal = df.where(col("failure") == 0)
    df_failure = df.where(col("failure") == 1)
    print(f"Number of normal records: {df_normal.count()}")
    print(f"Number of failure records: {df_failure.count()}")

    # Load the training pipeline
    pipeline_path = f"./pipelines/LSTM_AE_pipeline/pipeline_2024_12_bb_3d"
    if not os.path.exists(pipeline_path):
        print(f"Error: Pipeline not found at {pipeline_path}")
        return

    fitted_pipeline = PipelineModel.load(pipeline_path)
    print("Loaded training pipeline successfully")

    # Transform both normal and failure data using the training pipeline
    processed_normal = fitted_pipeline.transform(df_normal)
    processed_failure = fitted_pipeline.transform(df_failure)

    # Convert to numpy arrays
    normal_vectors = [
        row.features.toArray() for row in processed_normal.select("features").collect()
    ]
    failure_vectors = [
        row.features.toArray() for row in processed_failure.select("features").collect()
    ]

    normal_data = np.array(normal_vectors, dtype=np.float32)
    failure_data = np.array(failure_vectors, dtype=np.float32)

    # Create sequences
    print("\nCreating sequences...")
    X_normal_sequences = create_sequences(normal_data, timesteps)
    print(f"Created {len(X_normal_sequences)} normal sequences")

    # Create sequences for failure data that preserve all failure records
    if len(failure_data) > 0:
        print(f"Creating sequences for {len(failure_data)} failure records")
        X_failure_sequences = []

        # For each failure record, create a sequence centered around it
        for i in range(len(failure_data)):
            # Calculate start and end indices for the sequence
            start_idx = max(0, i - timesteps // 2)
            end_idx = min(len(failure_data), i + timesteps // 2)

            # If we don't have enough records before or after, adjust the sequence
            if end_idx - start_idx < timesteps:
                if start_idx == 0:
                    end_idx = min(len(failure_data), timesteps)
                else:
                    start_idx = max(0, end_idx - timesteps)

            # Create the sequence
            sequence = failure_data[start_idx:end_idx]

            # If sequence is shorter than timesteps, pad with normal data
            if len(sequence) < timesteps:
                padding_needed = timesteps - len(sequence)
                if start_idx == 0:  # Pad at the beginning
                    padding_data = normal_data[:padding_needed]
                    sequence = np.vstack([padding_data, sequence])
                else:  # Pad at the end
                    padding_data = normal_data[:padding_needed]
                    sequence = np.vstack([sequence, padding_data])

            X_failure_sequences.append(sequence)

        X_failure_sequences = np.array(X_failure_sequences)
        print(f"Created {len(X_failure_sequences)} failure sequences")
    else:
        X_failure_sequences = np.empty((0, timesteps, normal_data.shape[1]))
        print("No failure data to create sequences from")

    # Load the trained model
    model_path = f"./ml_models/LSTM_AE_{train_table_name}.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    lstm_ae_model = load_model(model_path, compile=False)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0005, clipnorm=1.0)
    lstm_ae_model.compile(optimizer=optimizer, loss="mae")

    # Load threshold
    threshold_path = f"./dataset/thresholds/LSTM_AE_threshold_{train_table_name}.txt"
    if not os.path.exists(threshold_path):
        print(f"Error: Threshold file not found at {threshold_path}")
        return

    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())

    # Make predictions
    print(f"\nMaking predictions...")
    print(f"X_normal_sequences shape: {X_normal_sequences.shape}")
    print(f"X_failure_sequences shape: {X_failure_sequences.shape}")

    reconstructed_normal = lstm_ae_model.predict(X_normal_sequences)
    print(f"Reconstructed normal shape: {reconstructed_normal.shape}")

    if len(X_failure_sequences) > 0:
        reconstructed_failure = lstm_ae_model.predict(X_failure_sequences)
        print(f"Reconstructed failure shape: {reconstructed_failure.shape}")
    else:
        print("No failure sequences to predict on")
        reconstructed_failure = np.array([])
        reconstruction_errors_failure = np.array([])
        return

    reconstruction_errors_normal = mae_error(X_normal_sequences, reconstructed_normal)
    reconstruction_errors_failure = mae_error(
        X_failure_sequences, reconstructed_failure
    )

    # Calculate metrics
    normal_anomalies = reconstruction_errors_normal > threshold
    failure_anomalies = reconstruction_errors_failure > threshold

    false_positives = np.sum(normal_anomalies)
    true_positives = np.sum(failure_anomalies)
    total_normal = len(reconstruction_errors_normal)
    total_failure = len(reconstruction_errors_failure)

    false_positive_rate = (
        (false_positives / total_normal) * 100 if total_normal > 0 else 0
    )
    detection_rate = (true_positives / total_failure) * 100 if total_failure > 0 else 0

    print("\nTest Results:")
    print(f"Anomaly threshold: {threshold:.4f}")
    print(f"Total normal sequences: {total_normal}")
    print(f"Total failure sequences: {total_failure}")
    print(f"False positives (normal sequences flagged as anomaly): {false_positives}")
    print(f"True positives (failure sequences correctly flagged): {true_positives}")
    print(f"False positive rate: {false_positive_rate:.2f}%")
    print(f"Detection rate: {detection_rate:.2f}%")

    # Generate ROC curve
    all_errors = np.concatenate(
        [reconstruction_errors_normal, reconstruction_errors_failure]
    )
    all_labels = np.concatenate([np.zeros(total_normal), np.ones(total_failure)])
    plot_roc_curve(all_labels, all_errors, test_table_name)

    # Save anomaly records
    anomaly_records = []
    added_original_indices = set()

    detected_failure_sequences_count = 0
    total_failure_sequences = len(reconstruction_errors_failure)

    if total_failure_sequences > 0:
        anomalous_sequence_indices = np.where(
            reconstruction_errors_failure > threshold
        )[0]
        detected_failure_sequences_count = len(anomalous_sequence_indices)

        for sequence_idx in anomalous_sequence_indices:
            # This sequence (sequence_idx) is anomalous.
            # It was formed from failure_data[sequence_idx] to failure_data[sequence_idx + timesteps_arg - 1]
            for j in range(timesteps):
                failure_record_inner_idx = sequence_idx + j
                if failure_record_inner_idx < len(df_failure.toPandas().index):
                    original_pdf_index = df_failure.toPandas().index[
                        failure_record_inner_idx
                    ]

                    if original_pdf_index not in added_original_indices:
                        record = (
                            df_failure.toPandas()
                            .iloc[failure_record_inner_idx]
                            .to_dict()
                        )
                        # Assign the reconstruction error of the sequence to each record from it
                        record["reconstruction_error_sequence"] = (
                            reconstruction_errors_failure[sequence_idx]
                        )
                        record["is_failure"] = (
                            True  # These records are from failure_data
                        )
                        anomaly_records.append(record)
                        added_original_indices.add(original_pdf_index)

    # Calculate false positive rate for normal test data
    false_positives_sequences = sum(
        1 for error in reconstruction_errors_normal if error > threshold
    )
    total_normal_sequences = len(reconstruction_errors_normal)
    false_positive_rate = (
        (false_positives_sequences / total_normal_sequences) * 100
        if total_normal_sequences > 0
        else 0
    )

    detection_rate_sequences = (
        (detected_failure_sequences_count / total_failure_sequences) * 100
        if total_failure_sequences > 0
        else 0
    )

    print(f"Total failure sequences presented to model: {total_failure_sequences}")
    print(f"Detected anomalous failure sequences: {detected_failure_sequences_count}")
    print(f"Sequence-based detection rate: {detection_rate_sequences:.2f}%")
    print(
        f"Total unique original failure records identified in anomalous sequences: {len(added_original_indices)}"
    )
    print(f"Total normal test sequences: {total_normal_sequences}")
    print(
        f"False positive sequences (normal sequences flagged as anomaly): {false_positives_sequences}"
    )
    print(f"False positive rate (sequences): {false_positive_rate:.2f}%")

    # Save anomaly records
    if anomaly_records:
        output_dir = "./dataset/anomalies"
        os.makedirs(output_dir, exist_ok=True)
        anomaly_df = pd.DataFrame(anomaly_records)
        anomaly_path = os.path.join(
            output_dir, f"LSTM_AE_test_anomalies_{test_table_name}.csv"
        )
        anomaly_df.to_csv(anomaly_path, index=False)
        print(f"\nAnomaly records saved to: {anomaly_path}")
    else:
        print("No unique original failure records identified in anomalous sequences.")


if __name__ == "__main__":
    test_table_name = "2024_12_15_test"
    test_lstm_ae_model(test_table_name)
