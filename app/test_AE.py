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
from pyspark.sql.functions import col, count, when, stddev_pop, udf, isnan
from pyspark.sql.types import NumericType, BooleanType
from lib.utils import (
    mae_error_ae,
    infer_column_types_from_schema,
    plot_roc_curve,
    boolean_columns,
)


def test_ae_model(test_table_name, train_table_name="2024_12_bb_3d"):
    """
    Test the Autoencoder model on a dataset containing both failure and non-failure records.
    """
    # Initialize Spark session
    spark = (
        SparkSession.builder.appName("TestAE")
        .master("local[*]")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    # Load the training dataset first to ensure we have the same schema
    print(f"Loading training dataset: {train_table_name}")
    train_df = spark.read.option("header", "true").csv(
        f"./dataset/{train_table_name}.csv",
        inferSchema=True,
    )

    # Load the test dataset
    print(f"Loading test dataset: {test_table_name}")
    test_df = spark.read.option("header", "true").csv(
        f"./dataset/{test_table_name}.csv",
        inferSchema=True,
    )
    print(f"Initial DataFrame row count: {test_df.count()}")
    print(f"Initial failure count in df: {test_df.where(col('failure') == 1).count()}")

    # Get the columns from training data
    train_columns = set(train_df.columns)
    test_columns = set(test_df.columns)

    # Find missing columns in test data
    missing_columns = train_columns - test_columns
    if missing_columns:
        print(f"Adding missing columns from training data: {missing_columns}")
        for col_name in missing_columns:
            test_df = test_df.withColumn(
                col_name, col("failure").cast("double") * 0.0
            )  # Initialize with zeros

    # Drop columns that are in test but not in training
    extra_columns = test_columns - train_columns
    if extra_columns:
        print(f"Dropping extra columns not in training data: {extra_columns}")
        test_df = test_df.drop(*extra_columns)

    # Drop columns that are entirely NULL in both datasets
    cols_to_drop = []
    for c_name in train_df.columns:
        non_null_count = train_df.where(col(c_name).isNotNull()).count()
        if non_null_count == 0:
            cols_to_drop.append(c_name)
            print(f"Column '{c_name}' contains only NULL values and will be dropped.")

    if cols_to_drop:
        test_df = test_df.drop(*cols_to_drop)
        train_df = train_df.drop(*cols_to_drop)
        print(f"Dropped columns with all NULLs: {cols_to_drop}")

    # Convert boolean columns to numeric type in both datasets
    boolean_cols = boolean_columns(train_df.schema)
    if boolean_cols:
        print(f"Converting boolean columns to numeric: {boolean_cols}")
        for column in boolean_cols:
            test_df = test_df.withColumn(column, col(column).cast("integer"))
            train_df = train_df.withColumn(column, col(column).cast("integer"))

    # Split test data into normal and failure
    df_normal = test_df.where(col("failure") == 0)
    df_failure = test_df.where(col("failure") == 1)
    print(f"Number of normal records: {df_normal.count()}")
    print(f"Number of failure records: {df_failure.count()}")

    # Load or create the training pipeline
    pipeline_path = f"./pipelines/AE_pipeline/test_pipeline_{train_table_name}"
    try:
        if os.path.exists(pipeline_path):
            print(f"Loading existing pipeline from {pipeline_path}")
            fitted_pipeline = PipelineModel.load(pipeline_path)
        else:
            raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")
    except Exception as e:
        print(f"Error loading pipeline: {str(e)}")
        print("Rebuilding pipeline...")
        # Create pipeline directory if it doesn't exist
        os.makedirs("./pipelines/AE_pipeline", exist_ok=True)

        # Infer schema & build feature pipeline using training data
        categorical_cols, numerical_cols = infer_column_types_from_schema(
            train_df.schema
        )

        # Identify numerical columns that are entirely null or NaN in training data
        problematic_numerical_cols = []
        for col_name in numerical_cols:
            if (
                train_df.where(
                    col(col_name).isNotNull() & ~isnan(col(col_name))
                ).count()
                == 0
            ):
                problematic_numerical_cols.append(col_name)
                print(
                    f"Warning: Numerical column '{col_name}' contains only null/NaN values in training data and will be excluded from imputation and features."
                )

        valid_numerical_cols = [
            c for c in numerical_cols if c not in problematic_numerical_cols
        ]

        # Process data in a single pipeline to minimize memory usage
        print("Building and fitting pipeline...")

        # Handle categorical columns
        indexers = [
            StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
            for c in categorical_cols
        ]

        # Handle numerical columns with imputation
        imputer = Imputer(
            inputCols=valid_numerical_cols,
            outputCols=valid_numerical_cols,
            strategy="mean",
        )

        # Combine all features
        feature_cols = (
            [c + "_idx" for c in categorical_cols] + valid_numerical_cols + boolean_cols
        )

        # Assemble features with proper handling of invalid values
        assembler = VectorAssembler(
            inputCols=feature_cols, outputCol="assembled_features", handleInvalid="keep"
        )

        # Scale features with robust scaling
        scaler = StandardScaler(
            inputCol="assembled_features",
            outputCol="features",
            withMean=True,
            withStd=True,
        )

        # Create and fit pipeline
        pipeline = Pipeline(stages=[imputer] + indexers + [assembler, scaler])
        fitted_pipeline = pipeline.fit(train_df)

        # Print feature information
        print("\nFeature Information:")
        print(f"Number of categorical features: {len(categorical_cols)}")
        print(f"Number of numerical features: {len(valid_numerical_cols)}")
        print(f"Number of boolean features: {len(boolean_cols)}")
        print(f"Total features after processing: {len(feature_cols)}")

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

    # Load the trained model
    model_path = f"./ml_models/AE_{train_table_name}.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    ae_model = load_model(model_path, compile=False)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    ae_model.compile(optimizer=optimizer, loss="mae")

    # Get model's expected input shape
    expected_input_shape = ae_model.input_shape[1:]
    print(f"Model expects input shape: {expected_input_shape}")
    print(f"Normal data shape before reshaping: {normal_data.shape}")
    print(f"Failure data shape before reshaping: {failure_data.shape}")

    # Check if there's a mismatch between model's expected input and actual data
    if normal_data.shape[1] != expected_input_shape[0]:
        print(f"\nWARNING: Feature dimension mismatch!")
        print(
            f"Model expects {expected_input_shape[0]} features but data has {normal_data.shape[1]} features"
        )
        print("This is likely due to columns being dropped during preprocessing.")
        print("Padding data with zeros to match expected input shape...")

        # Pad the data with zeros to match the expected input shape
        padding_size = expected_input_shape[0] - normal_data.shape[1]
        normal_data = np.pad(
            normal_data, ((0, 0), (0, padding_size)), mode="constant", constant_values=0
        )
        if len(failure_data) > 0:
            failure_data = np.pad(
                failure_data,
                ((0, 0), (0, padding_size)),
                mode="constant",
                constant_values=0,
            )

        print(
            f"New data shapes after padding: Normal={normal_data.shape}, Failure={failure_data.shape}"
        )

    # Load threshold
    threshold_path = f"./dataset/thresholds/AE_threshold_2024_12_bb_3d.txt"
    if not os.path.exists(threshold_path):
        print(f"Error: Threshold file not found at {threshold_path}")
        return

    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())
        print(f"\nLoaded threshold from {threshold_path}: {threshold:.4f}")

    # Make predictions
    print(f"\nMaking predictions...")

    # Process normal data in one batch
    print("Processing normal data...")
    reconstructed_normal = ae_model.predict(normal_data, verbose=0)
    print(f"Reconstructed normal shape: {reconstructed_normal.shape}")

    if len(failure_data) > 0:
        # Process failure data in one batch
        print("Processing failure data...")
        reconstructed_failure = ae_model.predict(failure_data, verbose=0)
        print(f"Reconstructed failure shape: {reconstructed_failure.shape}")
    else:
        print("No failure data to predict on")
        reconstructed_failure = np.array([])
        reconstruction_errors_failure = np.array([])
        return

    # Calculate reconstruction errors in one batch
    print("\nCalculating reconstruction errors...")
    reconstruction_errors_normal = mae_error_ae(normal_data, reconstructed_normal)
    reconstruction_errors_failure = mae_error_ae(failure_data, reconstructed_failure)

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
    print(f"Total normal records: {total_normal}")
    print(f"Total failure records: {total_failure}")
    print(f"False positives (normal records flagged as anomaly): {false_positives}")
    print(f"True positives (failure records correctly flagged): {true_positives}")
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

    # Process normal records flagged as anomalies
    if total_normal > 0:
        print("\nProcessing normal anomaly records...")
        anomalous_normal_indices = np.where(normal_anomalies)[0]
        print(f"Found {len(anomalous_normal_indices)} anomalous normal records")

        # Process all anomalous normal records without capping
        print("Processing all anomalous normal records...")
        # Convert normal data to pandas
        normal_df = df_normal.toPandas()
        for idx in anomalous_normal_indices:
            if (
                idx < len(normal_df)
                and normal_df.index[idx] not in added_original_indices
            ):
                record = normal_df.iloc[idx].to_dict()
                record["reconstruction_error"] = reconstruction_errors_normal[idx]
                record["is_failure"] = False
                anomaly_records.append(record)
                added_original_indices.add(normal_df.index[idx])

    # Process failure records flagged as anomalies
    if total_failure > 0:
        print("\nProcessing failure anomaly records...")
        anomalous_failure_indices = np.where(failure_anomalies)[0]
        print(f"Found {len(anomalous_failure_indices)} anomalous failure records")

        # Convert failure data to pandas
        failure_df = df_failure.toPandas()
        for idx in anomalous_failure_indices:
            if (
                idx < len(failure_df)
                and failure_df.index[idx] not in added_original_indices
            ):
                record = failure_df.iloc[idx].to_dict()
                record["reconstruction_error"] = reconstruction_errors_failure[idx]
                record["is_failure"] = True
                anomaly_records.append(record)
                added_original_indices.add(failure_df.index[idx])

    # Save anomaly records to CSV with additional metadata
    if anomaly_records:
        print("\nSaving anomaly records...")
        anomaly_df = pd.DataFrame(anomaly_records)

        # Add additional metadata columns
        anomaly_df["threshold"] = threshold
        anomaly_df["error_to_threshold_ratio"] = (
            anomaly_df["reconstruction_error"] / threshold
        )

        # Sort by reconstruction error in descending order
        anomaly_df = anomaly_df.sort_values("reconstruction_error", ascending=False)

        # Create anomalies directory if it doesn't exist
        os.makedirs("./dataset/anomalies", exist_ok=True)
        output_path = f"./dataset/anomalies/AE_test_anomalies_{test_table_name}.csv"
        anomaly_df.to_csv(output_path, index=False)
        print(f"\nSaved {len(anomaly_records)} anomaly records to {output_path}")
        print(f"  - {len(anomalous_normal_indices)} from normal records")
        print(f"  - {len(anomalous_failure_indices)} from failure records")

    else:
        print("\nNo anomalies detected in the dataset")

    # Clean up
    spark.stop()


if __name__ == "__main__":
    test_table_name = "2024_12_15_test"  # Replace with your test dataset name
    test_ae_model(test_table_name)
