import sys
import os
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import tensorflow as tf
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel  # Changed from Pipeline
from keras.models import load_model
from keras.optimizers.legacy import Adam
from pyspark.sql.functions import col, count, when, udf, lit
from pyspark.sql.types import BooleanType, IntegerType  # Added IntegerType for casting
from lib.utils import (
    create_sequences,
    mae_error,
    # infer_column_types_from_schema, # No longer inferring schema here
    plot_roc_curve,
    # boolean_columns, # Handled by saved pipeline or not needed if pre-cast
)

# The build_pipeline function is removed as we will load a pre-trained pipeline.


def get_sequence_labels(original_labels_series, num_sequences, timesteps):
    """
    Determines the label for each sequence. A sequence is a 'failure' (1)
    if any of its constituent original records were failures.
    Assumes original_labels_series corresponds to the data from which sequences were made.
    """
    sequence_labels = np.zeros(num_sequences, dtype=int)
    original_labels = (
        original_labels_series.to_numpy()
    )  # Convert pandas Series to numpy array

    if (
        len(original_labels) < timesteps
    ):  # Not enough data to form even one full sequence
        return sequence_labels

    for i in range(num_sequences):
        # The i-th sequence corresponds to original records from index i to i + timesteps - 1
        sequence_window = original_labels[i : i + timesteps]
        if np.any(sequence_window == 1):
            sequence_labels[i] = 1
    return sequence_labels


def generate_test_anomaly_report(
    reconstruction_errors_arg,
    threshold_arg,
    original_pdf_with_failures_arg,  # Pandas DF of the test data with original values and 'failure' column
    X_sequences_arg,  # The sequences themselves (features)
    y_true_sequence_labels_arg,  # Ground truth labels for each sequence
    timesteps_arg,
    test_table_name_arg,
):
    """
    Generates and saves an anomaly report for the test dataset.
    Lists unique original records from detected anomalous sequences.
    original_pdf_with_failures_arg should contain records corresponding to the data fed into create_sequences.
    """
    anomaly_records_list = []
    # This set will store the original index of records already added to the report
    # to ensure uniqueness of reported *original records*.
    added_original_record_indices = set()

    anomalous_sequence_indices = np.where(reconstruction_errors_arg > threshold_arg)[0]
    print(f"Found {len(anomalous_sequence_indices)} sequences with error > threshold.")

    # original_pdf_with_failures_arg is the pandas DF *before* create_sequences was called
    # It should have an index that allows us to retrieve original rows.
    # Let's assume original_pdf_with_failures_arg.index aligns with the data used for create_sequences.

    for seq_idx in anomalous_sequence_indices:
        # This sequence is anomalous.
        # It was formed from original records spanning from index 'seq_idx' to 'seq_idx + timesteps_arg - 1'
        # in the 'original_pdf_with_failures_arg' DataFrame.

        # We want to report each original record within this anomalous sequence, if not already reported.
        for j in range(timesteps_arg):
            original_record_index_in_pdf = seq_idx + j

            if original_record_index_in_pdf < len(original_pdf_with_failures_arg):
                if original_record_index_in_pdf not in added_original_record_indices:
                    try:
                        record_series = original_pdf_with_failures_arg.iloc[
                            original_record_index_in_pdf
                        ]
                        record_dict = record_series.to_dict()

                        # Add more info to the record
                        record_dict["sequence_index_where_found_anomalous"] = seq_idx
                        record_dict["position_in_sequence"] = j
                        record_dict["sequence_reconstruction_error"] = (
                            reconstruction_errors_arg[seq_idx]
                        )
                        record_dict["is_part_of_true_failure_sequence"] = bool(
                            y_true_sequence_labels_arg[seq_idx] == 1
                        )
                        record_dict["original_record_is_failure_ground_truth"] = bool(
                            record_series.get("failure", 0) == 1
                        )
                        record_dict["identified_as_anomaly_by_model"] = (
                            True  # Since seq_idx is in anomalous_sequence_indices
                        )

                        anomaly_records_list.append(record_dict)
                        added_original_record_indices.add(original_record_index_in_pdf)
                    except IndexError:
                        print(
                            f"Warning: Original record index {original_record_index_in_pdf} out of bounds for original_pdf_with_failures_arg."
                        )
            else:
                # This can happen if the last few sequences draw from the tail end of the data.
                pass

    if anomaly_records_list:
        output_dir = "./dataset/anomalies"
        os.makedirs(output_dir, exist_ok=True)
        anomaly_report_filename = os.path.join(
            output_dir, f"LSTM_AE_anomalies_TEST_{test_table_name_arg}.csv"
        )
        anomaly_df = pd.DataFrame(anomaly_records_list)

        # Reorder columns for clarity, putting key identifiers first
        cols_order = [
            "original_record_is_failure_ground_truth",
            "identified_as_anomaly_by_model",
            "sequence_reconstruction_error",
            "is_part_of_true_failure_sequence",
            "sequence_index_where_found_anomalous",
            "position_in_sequence",
        ]

        # Add original data columns, handling potential missing ones if original_pdf_with_failures_arg was minimal
        original_cols = [
            col
            for col in original_pdf_with_failures_arg.columns
            if col not in ["features", "assembled_features"] and col not in cols_order
        ]
        final_cols_order = cols_order + original_cols

        # Ensure all columns in final_cols_order exist in anomaly_df, add if not (e.g. if record_dict didn't have all original cols)
        for col_name in final_cols_order:
            if col_name not in anomaly_df.columns:
                anomaly_df[col_name] = pd.NA  # Or some other placeholder

        anomaly_df = anomaly_df.reindex(columns=final_cols_order)
        anomaly_df.to_csv(anomaly_report_filename, index=False)
        print(f"Anomaly report for test data saved to: {anomaly_report_filename}")
        print(
            f"Number of unique original records in anomaly report: {len(added_original_record_indices)}"
        )
    else:
        print("No anomalous sequences found in the test data based on the threshold.")


def test_lstm_ae_model(
    test_table_name, training_pipeline_table_name="2024_12_bb_3d", timesteps=20
):
    """
    Tests the LSTM AE model on an entire dataset, loading a pre-trained pipeline and model.
    Calculates True Positive Rate (TPR) and False Positive Rate (FPR).
    """
    spark = (
        SparkSession.builder.appName(f"TestLSTMAE_{test_table_name}")
        .master("local[*]")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config(
            "spark.sql.execution.arrow.pyspark.enabled", "true"
        )  # For toPandas efficiency
        .getOrCreate()
    )

    print(f"--- Testing LSTM AE Model ---")
    print(f"Test Dataset: {test_table_name}")
    print(f"Using Pipeline/Model trained on: {training_pipeline_table_name}")
    print(f"Timesteps: {timesteps}")

    # Paths
    pipeline_path = (
        f"./pipelines/LSTM_AE_pipeline/pipeline_{training_pipeline_table_name}"
    )
    model_path = f"./ml_models/LSTM_AE_{training_pipeline_table_name}.keras"
    threshold_path = (
        f"./dataset/thresholds/LSTM_AE_threshold_{training_pipeline_table_name}.txt"
    )

    # --- 1. Load Pre-trained Pipeline, Model, and Threshold ---
    if not os.path.exists(pipeline_path):
        print(
            f"Error: Pipeline file not found at {pipeline_path}. Ensure training script was run."
        )
        spark.stop()
        return
    print(f"Loading pre-trained Spark ML pipeline from: {pipeline_path}")
    fitted_pipeline = PipelineModel.load(pipeline_path)

    if not os.path.exists(model_path):
        print(
            f"Error: Model file not found at {model_path}. Ensure training script was run."
        )
        spark.stop()
        return
    print(f"Loading pre-trained Keras model from: {model_path}")
    lstm_ae_model = load_model(model_path, compile=False)
    optimizer = Adam(
        learning_rate=0.0005, clipnorm=1.0
    )  # Use same optimizer params as training
    lstm_ae_model.compile(optimizer=optimizer, loss="mae")
    # lstm_ae_model.summary()

    if not os.path.exists(threshold_path):
        print(
            f"Error: Threshold file not found at {threshold_path}. Ensure training script was run."
        )
        spark.stop()
        return
    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())
    print(f"Loaded anomaly threshold: {threshold:.4f}")

    # --- 2. Load and Preprocess Test Data ---
    print(f"Loading test dataset: {test_table_name}.csv")
    try:
        df = spark.read.option("header", "true").csv(
            f"./dataset/{test_table_name}.csv", inferSchema=True
        )
    except Exception as e:
        print(f"Error loading test dataset {test_table_name}.csv: {e}")
        spark.stop()
        return

    print(f"Initial test DataFrame row count: {df.count()}")
    if "failure" not in df.columns:
        print(
            "Error: 'failure' column not found in test dataset. Cannot calculate TPR/FPR."
        )
        spark.stop()
        return
    initial_failure_count = df.where(col("failure") == 1).count()
    print(f"Initial failure count in test df: {initial_failure_count}")

    # Keep a copy for the anomaly report with original values
    original_test_pdf_for_report = df.toPandas()

    # Drop columns that are entirely NULL
    cols_to_drop = [
        c_name
        for c_name in df.columns
        if df.where(col(c_name).isNotNull()).count() == 0
    ]
    if cols_to_drop:
        df = df.drop(*cols_to_drop)
        print(f"Dropped all-NULL columns from test data: {cols_to_drop}")

    # Apply the loaded pipeline
    print("Applying loaded pipeline to the test data...")
    processed_df = fitted_pipeline.transform(df)

    # Validate features (remove rows with NaN/Inf in 'features' vector)
    def validate_features(vector):
        if vector is None:
            return False
        arr = vector.toArray()
        return not (np.isnan(arr).any() or np.isinf(arr).any())

    validate_features_udf = udf(validate_features, BooleanType())

    valid_rows_df = processed_df.filter(validate_features_udf(col("features")))
    num_valid_rows = valid_rows_df.count()
    num_invalid_rows = processed_df.count() - num_valid_rows
    print(f"Number of valid rows after pipeline and UDF validation: {num_valid_rows}")
    if num_invalid_rows > 0:
        print(f"Number of rows removed due to NaN/Inf in features: {num_invalid_rows}")

    if num_valid_rows == 0:
        print("Error: No valid rows after preprocessing the test data. Exiting.")
        spark.stop()
        return

    # Select features and failure column, then convert to Pandas
    # Ensure 'failure' is cast to int for consistency in get_sequence_labels
    processed_test_pdf = valid_rows_df.select(
        "features", col("failure").cast(IntegerType()).alias("failure")
    ).toPandas()

    # If original_test_pdf_for_report was from the full `df`, and processed_test_pdf is from `valid_rows_df`,
    # their indices won't align directly if rows were dropped by UDF.
    # Let's get the original data for ONLY the valid rows.
    original_valid_rows_pdf = valid_rows_df.drop(
        "features", "assembled_features"
    ).toPandas()  # Contains original columns for valid rows

    print(
        f"Shape of processed_test_pdf (features and labels for sequences): {processed_test_pdf.shape}"
    )

    # --- 3. Create Sequences for the Entire Test Dataset ---
    if processed_test_pdf.empty or processed_test_pdf["features"].isnull().all():
        print("Error: No features to create sequences from after processing. Exiting.")
        spark.stop()
        return

    all_features_np = np.array(
        processed_test_pdf["features"].apply(lambda x: np.array(x.toArray())).tolist(),
        dtype=np.float32,
    )

    if all_features_np.ndim == 1:  # Handle case of single feature vector
        if (
            all_features_np.shape[0] > 0
            and timesteps > 0
            and all_features_np.shape[0]
            % (
                timesteps
                * (
                    all_features_np.shape[0] // timesteps
                    if all_features_np.shape[0] // timesteps > 0
                    else 1
                )
            )
            == 0
        ):  # Check if it's a flat array of multiple features that needs reshape
            # This heuristic is risky, better to ensure features are always 2D from pipeline
            try:
                potential_n_features = fitted_pipeline.stages[
                    -1
                ].getOutputCol()  # 'features'
                # Attempt to get numFeatures from the last stage (scaler) or assembler
                num_output_features = -1
                if hasattr(
                    fitted_pipeline.stages[-1], "numOutputFeatures"
                ):  # StandardScalerModel does not have it
                    pass  # num_output_features = fitted_pipeline.stages[-1].numOutputFeatures
                elif hasattr(
                    fitted_pipeline.stages[-2], "numOutputFeatures"
                ):  # VectorAssemblerModel does not have it directly
                    # For VectorAssembler, output feature count is length of inputCols to it
                    # This is getting too complex; the pipeline should output 2D feature vectors.
                    # For now, assume if 1D, it's a single sequence's features flattened.
                    pass

                # If all_features_np is 1D, it implies either a single record with multiple features,
                # or multiple records with a single feature each. create_sequences expects (num_records, num_features).
                # If it's (num_features,), reshape to (1, num_features)
                # This part needs robust handling based on what the pipeline guarantees.
                # Assuming pipeline outputs vector for each row, so all_features_np should be 2D.
                # If it became 1D, it means only one row of features.
                if len(processed_test_pdf) == 1:
                    all_features_np = all_features_np.reshape(1, -1)
                else:  # Multiple rows, but features came out 1D - this is an issue.
                    print(
                        "Warning: all_features_np is 1D for multiple rows. Sequence creation might fail or be incorrect."
                    )

            except Exception as e:
                print(
                    f"Warning: Could not reliably determine feature dimension for 1D all_features_np: {e}"
                )
        elif all_features_np.shape[0] == 0:
            print(
                "Error: all_features_np is empty after extraction. Cannot create sequences."
            )
            spark.stop()
            return

    if all_features_np.shape[0] < timesteps:
        print(
            f"Error: Not enough records ({all_features_np.shape[0]}) in test data to form even one sequence of {timesteps} timesteps."
        )
        spark.stop()
        return

    print(f"Shape of all_features_np (for create_sequences): {all_features_np.shape}")
    X_all_sequences = create_sequences(all_features_np, timesteps)
    print(f"Created {X_all_sequences.shape[0]} sequences from the test data.")

    if X_all_sequences.shape[0] == 0:
        print("Error: No sequences were created from the test data. Exiting.")
        spark.stop()
        return

    # --- 4. Determine True Labels for Sequences ---
    # processed_test_pdf['failure'] contains the failure status for each *original record* that went into sequences
    y_true_sequence_labels = get_sequence_labels(
        processed_test_pdf["failure"], X_all_sequences.shape[0], timesteps
    )

    # --- 5. Predict and Evaluate ---
    print("Predicting with loaded model...")
    reconstructed_sequences = lstm_ae_model.predict(X_all_sequences, verbose=0)
    reconstruction_errors = mae_error(X_all_sequences, reconstructed_sequences)

    # Evaluation metrics
    TP = np.sum((reconstruction_errors > threshold) & (y_true_sequence_labels == 1))
    FP = np.sum((reconstruction_errors > threshold) & (y_true_sequence_labels == 0))
    FN = np.sum((reconstruction_errors <= threshold) & (y_true_sequence_labels == 1))
    TN = np.sum((reconstruction_errors <= threshold) & (y_true_sequence_labels == 0))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    F1_score = 2 * (Precision * TPR) / (Precision + TPR) if (Precision + TPR) > 0 else 0

    print("\\n--- Test Set Evaluation Results ---")
    print(f"Total sequences evaluated: {len(X_all_sequences)}")
    print(
        f"Sequences from actual failures (ground truth positive sequences): {np.sum(y_true_sequence_labels == 1)}"
    )
    print(
        f"Sequences from actual normals (ground truth negative sequences): {np.sum(y_true_sequence_labels == 0)}"
    )

    print(f"True Positives (TP)  | Anomalous sequences correctly identified: {TP}")
    print(f"False Positives (FP) | Normal sequences incorrectly marked anomalous: {FP}")
    print(f"False Negatives (FN) | Anomalous sequences missed: {FN}")
    print(f"True Negatives (TN)  | Normal sequences correctly identified: {TN}")

    print(f"True Positive Rate (TPR / Recall / Sensitivity): {TPR:.4f} (Goal: > 0.75)")
    print(f"False Positive Rate (FPR): {FPR:.4f} (Goal: < 0.08)")
    print(f"Accuracy: {Accuracy:.4f}")
    print(f"Precision: {Precision:.4f}")
    print(f"F1-score: {F1_score:.4f}")

    if TPR > 0.75 and FPR < 0.08:
        print("Performance goals MET!")
    else:
        print("Performance goals NOT MET.")
        if TPR <= 0.75:
            print(f"  TPR is too low ({TPR:.4f} vs >0.75)")
        if FPR >= 0.08:
            print(f"  FPR is too high ({FPR:.4f} vs <0.08)")

    # --- 6. Generate Anomaly Report ---
    # Use original_valid_rows_pdf for generating the report, as it contains the original data for the rows that were actually sequenced.
    generate_test_anomaly_report(
        reconstruction_errors,
        threshold,
        original_valid_rows_pdf,  # This DF should align with processed_test_pdf used for sequencing
        X_all_sequences,
        y_true_sequence_labels,
        timesteps,
        test_table_name,
    )

    # --- 7. Plot ROC Curve ---
    if len(y_true_sequence_labels) > 0 and len(reconstruction_errors) > 0:
        plot_roc_curve(
            y_true_sequence_labels,
            reconstruction_errors,
            f"LSTM_AE_TEST_{test_table_name}",
        )
    else:
        print("Not enough data to plot ROC curve for the test set.")

    print("--- Test script finished. ---")
    spark.stop()


if __name__ == "__main__":
    # Configuration for the test
    test_data_filename = "2025_12_15_test"
    # This should be the table name used when the main LSTM_AE.py script was run for training
    # and saving the pipeline, model, and threshold.
    training_data_source_name = "2024_12_bb_3d"
    sequence_timesteps = 20  # Should match the timesteps used during training

    test_lstm_ae_model(
        test_table_name=test_data_filename,
        training_pipeline_table_name=training_data_source_name,
        timesteps=sequence_timesteps,
    )
