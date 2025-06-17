import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import tensorflow as tf
import numpy as np
import pandas as pd
from pyspark.sql import types
from pyspark.sql.functions import col, lit
from keras.models import load_model
from pyspark.sql.functions import col
from lib.connector import SparkToAWS
from lib.utils import (
    mae_error_ae,
    plot_roc_curve,
    boolean_columns,
    build_and_fit_feature_pipeline,
)


def test_ae_model(test_table_name, train_table_name="2024_12_bb_3d"):
    """
    Test the Autoencoder model on a dataset containing both failure and non-failure records.
    Uses a common utility function for preprocessing to ensure consistency.
    """
    connector = SparkToAWS()
    spark = connector.create_local_spark_session()

    # Load the full training dataset (used for schema reference and normal subset)
    print(f"Loading full training dataset: {train_table_name}")
    train_df_full = spark.read.option("header", "true").csv(
        f"./dataset/{train_table_name}.csv",
        inferSchema=True,
    )

    # Load the test dataset
    print(f"Loading test dataset: {test_table_name}")
    test_df = spark.read.option("header", "true").csv(
        f"./dataset/{test_table_name}.csv",
        inferSchema=True,
    )
    print(f"Initial test_df row count: {test_df.count()}")
    print(
        f"Initial failure count in test_df: {test_df.where(col('failure') == 1).count()}"
    )

    # --- Schema Alignment and Boolean Conversion (Consistent with AE_model.py) ---
    train_df_full_cols = set(train_df_full.columns)
    test_df_cols = set(test_df.columns)

    missing_in_test = train_df_full_cols - test_df_cols
    if missing_in_test:
        print(
            f"Adding missing columns to test_df from training data schema: {missing_in_test}"
        )
        for col_name in missing_in_test:
            train_col_type = train_df_full.schema[col_name].dataType
            test_df = test_df.withColumn(col_name, lit(None).cast(train_col_type))

    extra_in_test = test_df_cols - train_df_full_cols
    if extra_in_test:
        print(
            f"Dropping extra columns from test_df not in training data schema: {extra_in_test}"
        )
        test_df = test_df.drop(*extra_in_test)

    # Identify boolean columns from the *full* training schema (before filtering to normal)
    # These names will be passed to the utility function.
    boolean_col_names_from_full_train_schema = boolean_columns(train_df_full.schema)

    if boolean_col_names_from_full_train_schema:
        print(
            f"Casting boolean columns to integer in train_df_full and test_df: {boolean_col_names_from_full_train_schema}"
        )
        for column_name in boolean_col_names_from_full_train_schema:
            train_df_full = train_df_full.withColumn(
                column_name, col(column_name).cast(types.IntegerType())
            )
            test_df = test_df.withColumn(
                column_name, col(column_name).cast(types.IntegerType())
            )
    # --- End Schema Alignment and Boolean Conversion ---

    # Create the normal subset of training data for fitting the pipeline
    train_df_normal_subset = train_df_full.where(col("failure") == 0)
    print(
        f"Filtered training data to {train_df_normal_subset.count()} normal records for pipeline fitting."
    )

    # --- Use the common utility function to build and fit the pipeline ---
    # train_df_normal_subset already has boolean columns cast to int.
    # boolean_col_names_from_full_train_schema contains their original names.
    fitted_pipeline, feature_cols_from_util = build_and_fit_feature_pipeline(
        train_df_normal_subset, boolean_col_names_from_full_train_schema
    )
    # --- End of using utility function ---

    # Print information about the test dataset composition (after alignment and boolean conversion)
    print(f"\nTotal records in test_df for processing: {test_df.count()}")
    actual_normal_count = test_df.where(col("failure") == 0).count()
    actual_failure_count = test_df.where(col("failure") == 1).count()
    print(f"Actual normal records in test_df: {actual_normal_count}")
    print(f"Actual failure records in test_df: {actual_failure_count}")

    # Transform the entire test data using the common pipeline
    print("\nTransforming the entire test dataset using the common pipeline...")
    processed_test_df = fitted_pipeline.transform(test_df)

    collected_data = processed_test_df.select("features", "failure").collect()

    if not collected_data:
        print("No data after processing pipeline. Exiting.")
        spark.stop()
        return

    test_vectors = [row.features.toArray() for row in collected_data]
    original_labels = np.array([row.failure for row in collected_data]).astype(int)
    test_data_features = np.array(test_vectors, dtype=np.float32)

    print(f"Shape of extracted features: {test_data_features.shape}")
    print(f"Number of original labels extracted: {len(original_labels)}")

    model_path = f"./ml_models/AE_{train_table_name}.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        spark.stop()
        return

    ae_model = load_model(model_path, compile=False)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    ae_model.compile(optimizer=optimizer, loss="mae")

    expected_input_shape = ae_model.input_shape[1:]
    print(f"Model expects input shape: {expected_input_shape}")

    if (
        test_data_features.shape[0] > 0
        and test_data_features.shape[1] != expected_input_shape[0]
    ):
        print(f"\nWARNING: Feature dimension mismatch!")
        print(
            f"Model expects {expected_input_shape[0]} features, but data from common pipeline has {test_data_features.shape[1]} features."
        )
        print(
            f"Feature columns used by pipeline (total {len(feature_cols_from_util)}): {feature_cols_from_util}"
        )
        print(
            "This mismatch should ideally not happen if AE_model.py also uses the common utility correctly."
        )
        print(
            "Investigate AE_model.py's usage of build_and_fit_feature_pipeline if this occurs."
        )
        print(
            "Padding data with zeros to match expected input shape (may lead to inaccurate results)..."
        )
        padding_size = expected_input_shape[0] - test_data_features.shape[1]
        test_data_features = np.pad(
            test_data_features,
            ((0, 0), (0, padding_size)),
            mode="constant",
            constant_values=0,
        )
        print(f"New data shape after padding: {test_data_features.shape}")
    elif test_data_features.shape[0] == 0:
        print("No data features to process or check for padding.")

    threshold_path = f"./dataset/thresholds/AE_threshold_{train_table_name}.txt"
    if not os.path.exists(threshold_path):
        print(f"Error: Threshold file {threshold_path} not found.")
        # Fallback to default if specific train_table_name threshold doesn't exist
        fallback_threshold_path = f"./dataset/thresholds/AE_threshold_2024_12_bb_3d.txt"
        if (
            os.path.exists(fallback_threshold_path)
            and train_table_name != "2024_12_bb_3d"
        ):
            print(f"Attempting to use fallback threshold: {fallback_threshold_path}")
            threshold_path = fallback_threshold_path
        else:
            print(f"No suitable threshold file found. Exiting.")
            spark.stop()
            return

    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())
        print(f"\nLoaded threshold from {threshold_path}: {threshold:.4f}")

    if test_data_features.shape[0] > 0:
        print(f"\nMaking predictions on {test_data_features.shape[0]} test records...")
        reconstructed_test_data = ae_model.predict(test_data_features, verbose=0)
        print("\nCalculating reconstruction errors...")
        all_reconstruction_errors = mae_error_ae(
            test_data_features, reconstructed_test_data
        )

        # Print reconstruction error statistics
        print("\nReconstruction Error Statistics:")
        print(f"Mean error: {np.mean(all_reconstruction_errors):.4f}")
        print(f"Median error: {np.median(all_reconstruction_errors):.4f}")
        print(f"Standard deviation: {np.std(all_reconstruction_errors):.4f}")
        print(f"Min error: {np.min(all_reconstruction_errors):.4f}")
        print(f"Max error: {np.max(all_reconstruction_errors):.4f}")
        print(f"25th percentile: {np.percentile(all_reconstruction_errors, 25):.4f}")
        print(f"75th percentile: {np.percentile(all_reconstruction_errors, 75):.4f}")

        # --- Added for debugging reconstruction errors ---
        if len(original_labels) > 0 and len(all_reconstruction_errors) > 0:
            failure_indices = np.where(original_labels == 1)[0]
            normal_indices = np.where(original_labels == 0)[0]

            print(f"\nDEBUG: Reconstruction errors for actual FAILURES (label=1):")
            if len(failure_indices) > 0:
                for i, idx in enumerate(failure_indices):
                    print(
                        f"  Failure {i+1} (index {idx}): {all_reconstruction_errors[idx]:.4f}"
                    )
                    if i >= 9:  # Print first 10 failures
                        break
            else:
                print("  No failure records found in original_labels to debug.")

            print(
                f"\nDEBUG: Reconstruction errors for some actual NORMAL records (label=0):"
            )
            if len(normal_indices) > 0:
                for i, idx in enumerate(normal_indices[:5]):  # Print first 5 normal
                    print(
                        f"  Normal {i+1} (index {idx}): {all_reconstruction_errors[idx]:.4f}"
                    )
            else:
                print("  No normal records found in original_labels to debug.")
        # --- End of added debug section ---

    else:
        print("No data to make predictions on or calculate errors for.")
        all_reconstruction_errors = np.array([])
        original_labels = np.array([])  # Ensure consistency

    if len(original_labels) > 0 and len(all_reconstruction_errors) > 0:
        reconstruction_errors_normal = all_reconstruction_errors[original_labels == 0]
        reconstruction_errors_failure = all_reconstruction_errors[original_labels == 1]
    else:
        reconstruction_errors_normal = np.array([])
        reconstruction_errors_failure = np.array([])

    normal_anomalies = reconstruction_errors_normal > threshold
    failure_anomalies = reconstruction_errors_failure > threshold

    false_positives = np.sum(normal_anomalies)
    true_positives = np.sum(failure_anomalies)

    total_normal_from_labels = np.sum(original_labels == 0)
    total_failure_from_labels = np.sum(original_labels == 1)

    false_positive_rate = (
        (false_positives / total_normal_from_labels) * 100
        if total_normal_from_labels > 0
        else 0
    )
    detection_rate = (
        (true_positives / total_failure_from_labels) * 100
        if total_failure_from_labels > 0
        else 0
    )

    print("\nTest Results:")
    print(f"Anomaly threshold: {threshold:.4f}")
    print(f"Total normal records processed: {total_normal_from_labels}")
    print(f"Total failure records processed: {total_failure_from_labels}")
    print(f"False positives (normal records flagged as anomaly): {false_positives}")
    print(f"True positives (failure records correctly flagged): {true_positives}")
    print(f"False positive rate: {false_positive_rate:.2f}%")
    print(f"Detection rate: {detection_rate:.2f}%")

    if len(original_labels) > 0 and len(all_reconstruction_errors) > 0:
        plot_roc_curve(original_labels, all_reconstruction_errors, test_table_name)
    else:
        print("Not enough data to generate ROC curve.")

    anomaly_records = []
    if test_data_features.shape[0] > 0 and len(all_reconstruction_errors) > 0:
        test_df_pd = test_df.toPandas()
        for idx in range(len(all_reconstruction_errors)):
            if all_reconstruction_errors[idx] > threshold:
                if idx < len(test_df_pd):
                    record = test_df_pd.iloc[idx].to_dict()
                    record["reconstruction_error"] = all_reconstruction_errors[idx]
                    anomaly_records.append(record)

        if anomaly_records:
            print(
                f"\nFound {len(anomaly_records)} records with reconstruction error > threshold"
            )
        else:
            print("\nNo records found with reconstruction error > threshold.")
    else:
        print("\nNo data processed, skipping anomaly record saving.")

    if anomaly_records:
        print("\nSaving anomaly records...")
        anomaly_df = pd.DataFrame(anomaly_records)
        anomaly_df = anomaly_df.sort_values("reconstruction_error", ascending=False)
        os.makedirs("./dataset/anomalies", exist_ok=True)
        output_path = f"./dataset/anomalies/AE_test_anomalies_{test_table_name}.csv"
        anomaly_df.to_csv(output_path, index=False)
        print(f"\nSaved {len(anomaly_records)} anomaly records to {output_path}")
    else:
        print("\nNo anomalies detected or no data to process for saving.")

    connector.close_spark_session()


if __name__ == "__main__":
    test_table_name_arg = "2024_12_25_test"
    test_ae_model(test_table_name_arg)
