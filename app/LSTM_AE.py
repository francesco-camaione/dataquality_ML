import sys
import os
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer
from keras.models import Model, load_model
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from keras.optimizers.legacy import Adam
from lib.utils import (
    create_sequences,
    ms_error,
    infer_column_types_from_schema,
    plot_roc_curve,
)
from pyspark.sql.functions import col, count, when, stddev_pop, udf
from pyspark.sql.types import NumericType, BooleanType


def preprocess_data(spark, table_name_arg, timesteps_arg):
    """
    Loads, preprocesses, and splits the data.
    """
    df = spark.read.option("header", "true").csv(
        f"./dataset/{table_name_arg}.csv",
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
        print(
            f"Failure count after dropping all-NULL columns: {df.where(col('failure') == 1).count()}"
        )

    original_full_pdf_local = df.toPandas()

    categorical_cols, numerical_cols = infer_column_types_from_schema(df.schema)
    print(f"Original categorical columns: {categorical_cols}")
    print(f"Original numerical columns: {numerical_cols}")

    if "failure" in numerical_cols:
        numerical_cols.remove("failure")
        print(
            f"'failure' column removed from numerical_cols. New numerical_cols: {numerical_cols}"
        )

    indexers = [
        StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
        for c in categorical_cols
    ]
    feature_cols = [c + "_idx" for c in categorical_cols] + numerical_cols
    print(f"Final feature_cols for VectorAssembler: {feature_cols}")

    if not feature_cols:
        print(
            "Error: feature_cols is empty. No features to assemble and scale. Exiting."
        )
        sys.exit(1)

    problematic_numerical_cols = []
    if numerical_cols:
        df_schema = df.schema
        print(f"df_schema: {df_schema}")
        for nc in numerical_cols:
            field = df_schema[nc]
            data_type = field.dataType
            null_count_stats = df.select(
                count(when(col(nc).isNull(), nc)).alias("null_count")
            ).first()
            current_total_rows = df.count()
            is_all_null = null_count_stats["null_count"] == current_total_rows

            if is_all_null:
                print(f"Warning: Column '{nc}' (type: {data_type}) is ALL NULL.")
                problematic_numerical_cols.append(nc)
            elif isinstance(data_type, NumericType):
                stddev_stats = df.select(stddev_pop(nc).alias("stddev")).first()
                current_stddev = stddev_stats["stddev"]
                has_zero_stddev = current_stddev is not None and current_stddev == 0.0
                is_stddev_null = current_stddev is None

                if has_zero_stddev:
                    print(
                        f"Warning: Numerical column '{nc}' has zero standard deviation (all values are the same)."
                    )
                    problematic_numerical_cols.append(nc)
                elif is_stddev_null:
                    distinct_count = df.select(nc).distinct().count()
                    if distinct_count == 1:
                        print(
                            f"Warning: Numerical column '{nc}' has only one distinct value (implies zero standard deviation)."
                        )
                        problematic_numerical_cols.append(nc)
                    else:
                        print(
                            f"Warning: Numerical column '{nc}' has NULL standard deviation but is not all null and has {distinct_count} distinct values. Check data."
                        )
                        problematic_numerical_cols.append(nc)
            elif isinstance(data_type, BooleanType):
                distinct_non_null_count = df.select(nc).na.drop().distinct().count()
                if distinct_non_null_count <= 1:
                    print(
                        f"Warning: Boolean column '{nc}' has only one distinct non-null value or is all null (implies zero variance for scaling)."
                    )
                    problematic_numerical_cols.append(nc)

    if problematic_numerical_cols:
        print(
            f"\nFound problematic numerical/boolean columns: {problematic_numerical_cols}"
        )
        print(f"Removing them: {problematic_numerical_cols}")
        numerical_cols = [
            c for c in numerical_cols if c not in problematic_numerical_cols
        ]
        feature_cols = [c + "_idx" for c in categorical_cols] + numerical_cols
        print(
            f"Adjusted feature_cols after removing problematic columns: {feature_cols}"
        )
        if not feature_cols:
            print(
                "Error: feature_cols became empty after removing problematic columns. Exiting."
            )
            sys.exit(1)

    imputer_stages = []
    imputed_numerical_cols_final = numerical_cols
    if numerical_cols:
        print(f"Starting imputation for numerical columns: {numerical_cols}")
        imputer = Imputer(
            inputCols=numerical_cols,
            outputCols=[c + "_imputed" for c in numerical_cols],
            strategy="median",
        )
        imputer_stages.append(imputer)
        imputed_numerical_cols_final = [c + "_imputed" for c in numerical_cols]
        print(
            f"Numerical columns after imputation will be: {imputed_numerical_cols_final}"
        )

    feature_cols_for_assembler = [
        c + "_idx" for c in categorical_cols
    ] + imputed_numerical_cols_final

    def validate_features(vector):
        if vector is None:
            return False
        arr = vector.toArray()
        return not (np.isnan(arr).any() or np.isinf(arr).any())

    validate_features_udf = udf(validate_features, BooleanType())

    assembler = VectorAssembler(
        inputCols=feature_cols_for_assembler,
        outputCol="assembled_features",
        handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol="assembled_features", outputCol="features", withMean=True, withStd=True
    )
    pipeline_stages = indexers + imputer_stages + [assembler, scaler]
    feature_pipeline = Pipeline(stages=pipeline_stages)
    fitted_feature_pipeline = feature_pipeline.fit(df)
    processed_df_features = fitted_feature_pipeline.transform(df)
    print(
        f"Failure count after Spark ML pipeline (before UDF filter): {processed_df_features.where(col('failure') == 1).count()}"
    )

    valid_rows = processed_df_features.filter(validate_features_udf(col("features")))
    print(f"Number of valid rows after processing: {valid_rows.count()}")
    print(
        f"Failure count in valid_rows (after UDF filter): {valid_rows.where(col('failure') == 1).count()}"
    )
    print(
        f"Number of invalid rows: {processed_df_features.count() - valid_rows.count()}"
    )

    if valid_rows.count() == 0:
        print("Error: No valid rows after processing. Check data preprocessing steps.")
        sys.exit(1)

    processed_pdf = valid_rows.select("features", "failure").toPandas()
    print(
        f"Failure count in processed_pdf (Pandas): {processed_pdf[processed_pdf['failure'] == 1].shape[0]}"
    )
    normal_pdf = processed_pdf[processed_pdf["failure"] == 0].copy()
    failure_pdf = processed_pdf[processed_pdf["failure"] == 1].copy()
    print(f"Number of failure records in failure_pdf (Pandas): {failure_pdf.shape[0]}")
    failure_pdf_original_indices_local = failure_pdf.index.tolist()

    normal_feature_list = (
        normal_pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
    )
    normal_data = np.array(normal_feature_list, dtype=np.float32)
    failure_feature_list = (
        failure_pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
    )
    failure_data = np.array(failure_feature_list, dtype=np.float32)
    print(
        f"Shape of failure_data (NumPy before create_sequences): {failure_data.shape}"
    )

    X_normal_sequences = create_sequences(normal_data, timesteps_arg)
    # original_indices_normal_local = normal_pdf.index[
    # timesteps_arg - 1 : len(normal_pdf) - timesteps_arg + 1 + timesteps_arg - 1
    # ] # This was unused

    X_failure_sequences = create_sequences(failure_data, timesteps_arg)
    print(
        f"Shape of X_failure_sequences (NumPy after create_sequences): {X_failure_sequences.shape}"
    )
    corresponding_failure_pdf_indices_local = []
    if len(X_failure_sequences) > 0:
        for i in range(len(failure_data) - timesteps_arg + 1):
            corresponding_failure_pdf_indices_local.append(
                failure_pdf_original_indices_local[i]
            )

    if X_normal_sequences.ndim > 1 and X_normal_sequences.shape[0] > 0:
        train_size_normal = int(0.8 * len(X_normal_sequences))
        X_train_local = X_normal_sequences[:train_size_normal]
        X_test_normal_local = X_normal_sequences[train_size_normal:]
        y_test_normal_labels_local = np.zeros(len(X_test_normal_local))
    else:
        X_train_local = np.empty(
            (
                0,
                timesteps_arg,
                (
                    normal_data.shape[1]
                    if normal_data.ndim > 1 and normal_data.shape[1] > 0
                    else 0
                ),
            )
        )
        X_test_normal_local = np.empty(
            (
                0,
                timesteps_arg,
                (
                    normal_data.shape[1]
                    if normal_data.ndim > 1 and normal_data.shape[1] > 0
                    else 0
                ),
            )
        )
        y_test_normal_labels_local = np.array([])

    X_test_failure_local = X_failure_sequences
    y_test_failure_labels_local = np.ones(len(X_test_failure_local))

    print(f"X_train shape: {X_train_local.shape}")
    print(f"X_test_normal shape: {X_test_normal_local.shape}")
    print(f"X_test_failure shape: {X_test_failure_local.shape}")

    return (
        original_full_pdf_local,
        X_train_local,
        X_test_normal_local,
        X_test_failure_local,
        y_test_normal_labels_local,
        y_test_failure_labels_local,
        failure_pdf_original_indices_local,
        corresponding_failure_pdf_indices_local,
    )


def load_or_create_model(
    model_path_arg, timesteps_arg, n_features_arg, X_train_arg, table_name_arg
):
    """
    Loads an existing model or creates, trains, and saves a new one.
    """
    model_dir = "./ml_models"
    os.makedirs(model_dir, exist_ok=True)
    full_model_path = os.path.join(model_dir, f"LSTM_AE_{table_name_arg}.keras")

    lstm_ae_model = None
    if os.path.exists(full_model_path):
        print(f"Loading existing model from {full_model_path}")
        lstm_ae_model = load_model(full_model_path, compile=False)
        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
        lstm_ae_model.compile(optimizer=optimizer, loss="mse")
        print("Model loaded and compiled successfully.")
    else:
        print("Defining and training a new model...")
        input_seq = Input(shape=(timesteps_arg, n_features_arg))
        encoded = LSTM(32, activation="relu", return_sequences=True)(input_seq)
        encoded = LSTM(16, activation="relu")(encoded)
        decoded = RepeatVector(timesteps_arg)(encoded)
        decoded = LSTM(16, activation="relu", return_sequences=True)(decoded)
        decoded = LSTM(32, activation="relu", return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(n_features_arg))(decoded)
        lstm_ae_model = Model(inputs=input_seq, outputs=decoded)

        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
        lstm_ae_model.compile(optimizer=optimizer, loss="mse")
        lstm_ae_model.summary()

        if np.any(np.isnan(X_train_arg)) or np.any(np.isinf(X_train_arg)):
            print(
                "\nERROR: X_train contains NaNs or Infs before model training! Cannot proceed."
            )
            sys.exit(1)

        print("\nNo NaNs or Infs found in X_train. Proceeding with training.")
        print(
            f"Input to lstm_ae.fit - X_train shape: {X_train_arg.shape}, y_train shape: {X_train_arg.shape}"
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            full_model_path,
            monitor="val_loss",
            save_best_only=True,
            save_format="keras",
        )
        history = lstm_ae_model.fit(
            X_train_arg,
            X_train_arg,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            shuffle=True,
            callbacks=[early_stopping, model_checkpoint],
        )
        print(f"New model saved to {full_model_path}")
    return lstm_ae_model


def predict_and_evaluate(model, X_test_normal_arg, X_test_failure_arg):
    """
    Performs predictions and evaluates reconstruction errors.
    """
    X_test_normal_pred = model.predict(X_test_normal_arg, verbose=1)
    if np.any(np.isnan(X_test_normal_pred)):
        print("\nWARNING: Predictions on normal test data contain NaNs!")
        print(
            f"Number of NaNs in X_test_normal_pred: {np.sum(np.isnan(X_test_normal_pred))}"
        )
        X_test_normal_pred = np.nan_to_num(X_test_normal_pred, nan=0.0)
    else:
        print("\nNo NaNs found in X_test_normal_pred.")

    reconstruction_errors_normal_local = ms_error(X_test_normal_arg, X_test_normal_pred)
    if np.any(np.isnan(reconstruction_errors_normal_local)):
        reconstruction_errors_normal_local = np.nan_to_num(
            reconstruction_errors_normal_local, nan=0.0
        )

    if X_test_failure_arg.shape[0] > 0:
        X_test_failure_pred = model.predict(X_test_failure_arg, verbose=1)
        if np.any(np.isnan(X_test_failure_pred)):
            print("\nWARNING: Predictions on failure test data contain NaNs!")
            print(
                f"Number of NaNs in X_test_failure_pred: {np.sum(np.isnan(X_test_failure_pred))}"
            )
            X_test_failure_pred = np.nan_to_num(X_test_failure_pred, nan=0.0)
        else:
            print("\nNo NaNs found in X_test_failure_pred.")

        reconstruction_errors_failure_local = ms_error(
            X_test_failure_arg, X_test_failure_pred
        )
        if np.any(np.isnan(reconstruction_errors_failure_local)):
            print("\nWARNING: Reconstruction errors for failure data contain NaNs!")
            reconstruction_errors_failure_local = np.nan_to_num(
                reconstruction_errors_failure_local, nan=0.0
            )
    else:
        reconstruction_errors_failure_local = np.array([])

    return reconstruction_errors_normal_local, reconstruction_errors_failure_local


def generate_anomaly_report(
    reconstruction_errors_normal_arg,
    reconstruction_errors_failure_arg,
    threshold_arg,
    original_full_pdf_arg,
    corresponding_failure_pdf_indices_arg,
    table_name_arg,
):
    """
    Generates and saves the anomaly report.
    """
    if len(reconstruction_errors_failure_arg) > 0:
        detected_failures = sum(
            1 for error in reconstruction_errors_failure_arg if error > threshold_arg
        )
        total_failures = len(reconstruction_errors_failure_arg)
        detection_rate = (
            (detected_failures / total_failures) * 100 if total_failures > 0 else 0
        )
        false_positives = sum(
            1 for error in reconstruction_errors_normal_arg if error > threshold_arg
        )
        total_normal = len(reconstruction_errors_normal_arg)
        false_positive_rate = (
            (false_positives / total_normal) * 100 if total_normal > 0 else 0
        )

        print(f"Total failure sequences: {total_failures}")
        print(f"Detected failures: {detected_failures}")
        print(f"Detection rate: {detection_rate:.2f}%")
        print(f"Total normal test sequences: {total_normal}")
        print(f"False positives: {false_positives}")
        print(f"False positive rate: {false_positive_rate:.2f}%")
        print(f"Detection Rate Analysis for {table_name_arg}\n")
        print(f"Threshold: {threshold_arg:.4f}\n")
        print(f"Total failure sequences: {total_failures}\n")
        print(f"Detected failures: {detected_failures}\n")
        print(f"Detection rate: {detection_rate:.2f}%\n")
        print(f"Total normal test sequences: {total_normal}\n")
        print(f"False positives: {false_positives}\n")
        print(f"False positive rate: {false_positive_rate:.2f}%\n")

        output_dir = "./dataset/anomalies"
        os.makedirs(output_dir, exist_ok=True)
        anomaly_records_path = os.path.join(
            output_dir, f"LSTM_AE_anomalies_{table_name_arg}.csv"
        )
        anomaly_indices_failure = np.where(
            reconstruction_errors_failure_arg > threshold_arg
        )[0]
        anomaly_records = []

        for idx in anomaly_indices_failure:
            original_idx = corresponding_failure_pdf_indices_arg[idx]
            record = original_full_pdf_arg.iloc[original_idx].to_dict()
            record["reconstruction_error"] = reconstruction_errors_failure_arg[idx]
            record["is_failure"] = True
            anomaly_records.append(record)

        anomaly_df = pd.DataFrame(anomaly_records)
        anomaly_df.to_csv(anomaly_records_path, index=False)
        print(f"Failure anomaly records saved to: {anomaly_records_path}")


def main():
    table_name_main = "2024_12_bb_3d"
    timesteps_main = 20

    spark_main = (
        SparkSession.builder.appName("App")
        .master("local[*]")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    (
        original_full_pdf_main,
        X_train_main,
        X_test_normal_main,
        X_test_failure_main,
        y_test_normal_labels_main,
        y_test_failure_labels_main,
        _,
        corresponding_failure_pdf_indices_main,
    ) = preprocess_data(spark_main, table_name_main, timesteps_main)

    if X_train_main.shape[0] == 0 or X_train_main.shape[2] == 0:
        print(
            "Error: X_train is empty or has no features. Cannot proceed with model training/loading."
        )
        sys.exit(1)

    n_features_main = X_train_main.shape[2]
    model_path_main = os.path.join("./ml_models", f"LSTM_AE_{table_name_main}.keras")

    lstm_ae_main = load_or_create_model(
        model_path_main, timesteps_main, n_features_main, X_train_main, table_name_main
    )

    (
        reconstruction_errors_normal_main,
        reconstruction_errors_failure_main,
    ) = predict_and_evaluate(lstm_ae_main, X_test_normal_main, X_test_failure_main)

    all_reconstruction_errors_main = np.concatenate(
        [reconstruction_errors_normal_main, reconstruction_errors_failure_main]
    )
    all_true_labels_main = np.concatenate(
        [y_test_normal_labels_main, y_test_failure_labels_main]
    )

    threshold_main = 1.0
    if len(reconstruction_errors_normal_main) > 0:
        threshold_main = np.percentile(reconstruction_errors_normal_main, 95)
        print(
            f"\nAnomaly Threshold (95th percentile of normal data): {threshold_main:.4f}"
        )

    generate_anomaly_report(
        reconstruction_errors_normal_main,
        reconstruction_errors_failure_main,
        threshold_main,
        original_full_pdf_main,
        corresponding_failure_pdf_indices_main,
        table_name_main,
    )

    plot_roc_curve(
        all_true_labels_main, all_reconstruction_errors_main, table_name_main
    )

    print("Script finished.")


if __name__ == "__main__":
    main()
