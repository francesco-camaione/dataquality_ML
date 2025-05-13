import sys
import os

# --- Force TensorFlow to use CPU ---
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
# ---

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
    plot_reconstruction_error,
    infer_column_types_from_schema,
)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, count, when, stddev_pop, udf
from pyspark.sql.types import NumericType, BooleanType, BooleanType as SparkBooleanType

table_name = "2024_12_bb_3d"
spark = (
    SparkSession.builder.appName("App")
    .master("local[*]")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

df = spark.read.option("header", "true").csv(
    f"./dataset/{table_name}.csv",
    inferSchema=True,
)
print(f"Initial DataFrame row count: {df.count()}")

# --- Start: Drop columns that are entirely NULL ---
cols_to_drop = []
for c_name in df.columns:
    # Count non-null values for the current column
    non_null_count = df.where(col(c_name).isNotNull()).count()
    if non_null_count == 0:
        cols_to_drop.append(c_name)
        print(f"Column '{c_name}' contains only NULL values and will be dropped.")

if cols_to_drop:
    df = df.drop(*cols_to_drop)
    print(f"Dropped columns with all NULLs: {cols_to_drop}")
# --- End: Drop columns that are entirely NULL ---

original_full_pdf = df.toPandas()

# Assuming 'failure' column exists and is 0 for normal, 1 for failure.
# It should be identified as a numerical column by infer_column_types_from_schema.
categorical_cols, numerical_cols = infer_column_types_from_schema(df.schema)

print(f"Original categorical columns: {categorical_cols}")
print(f"Original numerical columns: {numerical_cols}")

# Ensure 'failure' is not in feature_cols if it's there by mistake
if "failure" in numerical_cols:
    numerical_cols.remove("failure")
    print(
        f"'failure' column removed from numerical_cols. New numerical_cols: {numerical_cols}"
    )
if "failure" in categorical_cols:  # Should not happen if it's 0/1
    categorical_cols.remove("failure")
    print(
        f"Warning: 'failure' column was treated as categorical and removed. New categorical_cols: {categorical_cols}"
    )

# maps strings to numbers
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in categorical_cols
]
feature_cols = [c + "_idx" for c in categorical_cols] + numerical_cols
print(f"Final feature_cols for VectorAssembler: {feature_cols}")

if not feature_cols:
    print("Error: feature_cols is empty. No features to assemble and scale. Exiting.")
    sys.exit(1)

problematic_numerical_cols = []
if numerical_cols:  # Only check if there are numerical columns
    # Get schema once for type lookup
    df_schema = df.schema
    for nc in numerical_cols:
        # Get the actual data type of the column
        field = df_schema[nc]
        data_type = field.dataType

        null_count_stats = df.select(
            count(when(col(nc).isNull(), nc)).alias("null_count")
        ).first()
        # Efficiently get total_rows once if not already available and df is large
        # For this loop, df.count() inside is acceptable for clarity, but for very large dfs consider getting it once outside
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
            if (
                distinct_non_null_count <= 1
            ):  # if 0 (all nulls, caught above) or 1 distinct non-null
                print(
                    f"Warning: Boolean column '{nc}' has only one distinct non-null value or is all null (implies zero variance for scaling)."
                )
                problematic_numerical_cols.append(nc)

if problematic_numerical_cols:
    print(
        f"\nFound problematic numerical/boolean columns: {problematic_numerical_cols}"
    )
    print(f"Removing them: {problematic_numerical_cols}")
    numerical_cols = [c for c in numerical_cols if c not in problematic_numerical_cols]
    # Reconstruct feature_cols after modifying numerical_cols
    feature_cols = [c + "_idx" for c in categorical_cols] + numerical_cols
    print(f"Adjusted feature_cols after removing problematic columns: {feature_cols}")
    if not feature_cols:
        print(
            "Error: feature_cols became empty after removing problematic columns. Exiting."
        )
        sys.exit(1)


# --- Start: Impute NaNs in remaining numerical columns ---
imputer_stages = []
imputed_numerical_cols_final = numerical_cols  # Start with current numerical_cols
if numerical_cols:  # Only impute if there are numerical columns
    print(f"Starting imputation for numerical columns: {numerical_cols}")
    imputer = Imputer(
        inputCols=numerical_cols,
        outputCols=[c + "_imputed" for c in numerical_cols],
        strategy="median",  # Changed from mean to median for better handling of outliers
    )
    imputer_stages.append(imputer)
    imputed_numerical_cols_final = [c + "_imputed" for c in numerical_cols]
    print(f"Numerical columns after imputation will be: {imputed_numerical_cols_final}")
# Update feature_cols to use imputed numerical columns
feature_cols_for_assembler = [
    c + "_idx" for c in categorical_cols
] + imputed_numerical_cols_final
print(
    f"Feature columns for VectorAssembler after imputation: {feature_cols_for_assembler}"
)
# --- End: Impute NaNs ---


# Add data validation after imputation
def validate_features(vector):
    if vector is None:
        return False
    arr = vector.toArray()
    return not (np.isnan(arr).any() or np.isinf(arr).any())


validate_features_udf = udf(validate_features, BooleanType())

# combines the features values all into a single vector
assembler = VectorAssembler(
    inputCols=feature_cols_for_assembler,
    outputCol="assembled_features",
    handleInvalid="keep",
)

scaler = StandardScaler(
    inputCol="assembled_features", outputCol="features", withMean=True, withStd=True
)

# Pipeline for feature processing
pipeline_stages = indexers + imputer_stages + [assembler, scaler]
feature_pipeline = Pipeline(stages=pipeline_stages)

fitted_feature_pipeline = feature_pipeline.fit(df)
processed_df_features = fitted_feature_pipeline.transform(df)

# Validate features after processing
valid_rows = processed_df_features.filter(validate_features_udf(col("features")))
print(f"Number of valid rows after processing: {valid_rows.count()}")
print(f"Number of invalid rows: {processed_df_features.count() - valid_rows.count()}")

if valid_rows.count() == 0:
    print("Error: No valid rows after processing. Check data preprocessing steps.")
    sys.exit(1)

# Select features and the failure column, then convert to Pandas
processed_pdf = valid_rows.select("features", "failure").toPandas()

# Separate normal and failure data
normal_pdf = processed_pdf[processed_pdf["failure"] == 0].copy()
failure_pdf = processed_pdf[processed_pdf["failure"] == 1].copy()

# Store original indices from failure_pdf which can be used to loc into original_full_pdf
failure_pdf_original_indices = failure_pdf.index.tolist()

# Prepare normal data features
normal_feature_list = (
    normal_pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
)
normal_data = np.array(normal_feature_list, dtype=np.float32)

# Prepare failure data features
failure_feature_list = (
    failure_pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
)
failure_data = np.array(failure_feature_list, dtype=np.float32)


# Ensure the data is reshaped correctly for LSTM
timesteps = 20

# Create sequences for normal data
X_normal_sequences = create_sequences(normal_data, timesteps)
# Keep track of original indices for normal data for later reference if needed for false positives
original_indices_normal = normal_pdf.index[
    timesteps - 1 : len(normal_pdf) - timesteps + 1 + timesteps - 1
]


# Create sequences for failure data
X_failure_sequences = create_sequences(failure_data, timesteps)
# Map sequence index back to the original index in failure_pdf, then to original_full_pdf
# These are indices within the `failure_pdf` dataframe.
corresponding_failure_pdf_indices = []
if len(X_failure_sequences) > 0:
    for i in range(len(failure_data) - timesteps + 1):
        corresponding_failure_pdf_indices.append(
            failure_pdf_original_indices[i]
        )  # Storing the original index from failure_pdf


print(f"Normal data shape: {normal_data.shape}")
print(
    f"Normal sequence shape: {X_normal_sequences.shape if X_normal_sequences.ndim > 1 else (0,)}"
)
print(f"Failure data shape: {failure_data.shape}")
print(
    f"Failure sequence shape: {X_failure_sequences.shape if X_failure_sequences.ndim > 1 else (0,)}"
)


# Split normal sequences into training and testing sets (e.g., 80% training for normal data)
if X_normal_sequences.ndim > 1 and X_normal_sequences.shape[0] > 0:
    train_size_normal = int(0.8 * len(X_normal_sequences))
    X_train = X_normal_sequences[:train_size_normal]
    X_test_normal = X_normal_sequences[train_size_normal:]
    y_test_normal_labels = np.zeros(len(X_test_normal))
else:  # Handle case with insufficient normal data
    X_train = np.empty(
        (
            0,
            timesteps,
            (
                normal_data.shape[1]
                if normal_data.ndim > 1 and normal_data.shape[1] > 0
                else 0
            ),
        )
    )
    X_test_normal = np.empty(
        (
            0,
            timesteps,
            (
                normal_data.shape[1]
                if normal_data.ndim > 1 and normal_data.shape[1] > 0
                else 0
            ),
        )
    )
    y_test_normal_labels = np.array([])

# Test set for failures
X_test_failure = X_failure_sequences
y_test_failure_labels = np.ones(len(X_test_failure))

print(f"X_train shape: {X_train.shape}")
print(f"X_test_normal shape: {X_test_normal.shape}")
print(f"X_test_failure shape: {X_test_failure.shape}")

if X_train.shape[0] == 0 or X_train.shape[2] == 0:
    print("Not enough data to train the model. Exiting.")
    sys.exit()

n_features = X_train.shape[2]
model_dir = "./ml_models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"LSTM_AE_{table_name}.keras")

lstm_ae = None
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    try:
        # Load model with custom_objects to handle optimizer
        lstm_ae = load_model(model_path, compile=False)
        # Re-compile the model with the same optimizer settings
        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
        lstm_ae.compile(optimizer=optimizer, loss="mse")
        print("Model loaded and compiled successfully.")
        lstm_ae.summary()
    except Exception as e:
        print(f"Error loading or compiling model: {e}. Will retrain.")
        lstm_ae = None

if lstm_ae is None:
    print("Defining and training a new model...")

    input_seq = Input(shape=(timesteps, n_features))
    encoded = LSTM(32, activation="relu", return_sequences=True)(input_seq)
    encoded = LSTM(16, activation="relu")(encoded)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(16, activation="relu", return_sequences=True)(decoded)
    decoded = LSTM(32, activation="relu", return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(n_features))(decoded)
    lstm_ae = Model(inputs=input_seq, outputs=decoded)

    # Compile with more robust optimizer settings
    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
    lstm_ae.compile(optimizer=optimizer, loss="mse")
    lstm_ae.summary()

    # Check for NaNs/Infs in X_train before fitting
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print(
            "\nERROR: X_train contains NaNs or Infs before model training! Cannot proceed."
        )
        print(f"Number of NaNs in X_train: {np.sum(np.isnan(X_train))}")
        print(f"Number of Infs in X_train: {np.sum(np.isinf(X_train))}")
        sys.exit(1)

    print("\nNo NaNs or Infs found in X_train. Proceeding with training.")
    print(
        f"Input to lstm_ae.fit - X_train shape: {X_train.shape}, y_train shape: {X_train.shape}"
    )

    # Add early stopping and model checkpointing
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor="val_loss", save_best_only=True, save_format="keras"
    )

    history = lstm_ae.fit(
        X_train,
        X_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        shuffle=True,
        callbacks=[early_stopping, model_checkpoint],
    )
    print(f"New model saved to {model_path}")

# Predict on normal test data with error handling
try:
    X_test_normal_pred = lstm_ae.predict(X_test_normal, verbose=1)
    if np.any(np.isnan(X_test_normal_pred)):
        print("\nWARNING: Predictions on normal test data contain NaNs!")
        print(
            f"Number of NaNs in X_test_normal_pred: {np.sum(np.isnan(X_test_normal_pred))}"
        )
        # Replace NaNs with zeros or another appropriate value
        X_test_normal_pred = np.nan_to_num(X_test_normal_pred, nan=0.0)
    else:
        print("\nNo NaNs found in X_test_normal_pred.")
except Exception as e:
    print(f"Error during prediction on normal test data: {e}")
    sys.exit(1)

# Calculate reconstruction errors with error handling
try:
    reconstruction_errors_normal = ms_error(X_test_normal, X_test_normal_pred)
    if np.any(np.isnan(reconstruction_errors_normal)):
        print("\nWARNING: Reconstruction errors for normal data contain NaNs!")
        reconstruction_errors_normal = np.nan_to_num(
            reconstruction_errors_normal, nan=0.0
        )
except Exception as e:
    print(f"Error calculating reconstruction errors for normal data: {e}")
    sys.exit(1)

# Predict on failure test data with error handling
if X_test_failure.shape[0] > 0:
    try:
        X_test_failure_pred = lstm_ae.predict(X_test_failure, verbose=1)
        if np.any(np.isnan(X_test_failure_pred)):
            print("\nWARNING: Predictions on failure test data contain NaNs!")
            print(
                f"Number of NaNs in X_test_failure_pred: {np.sum(np.isnan(X_test_failure_pred))}"
            )
            X_test_failure_pred = np.nan_to_num(X_test_failure_pred, nan=0.0)
        else:
            print("\nNo NaNs found in X_test_failure_pred.")
    except Exception as e:
        print(f"Error during prediction on failure test data: {e}")
        sys.exit(1)

    try:
        reconstruction_errors_failure = ms_error(X_test_failure, X_test_failure_pred)
        if np.any(np.isnan(reconstruction_errors_failure)):
            print("\nWARNING: Reconstruction errors for failure data contain NaNs!")
            reconstruction_errors_failure = np.nan_to_num(
                reconstruction_errors_failure, nan=0.0
            )
    except Exception as e:
        print(f"Error calculating reconstruction errors for failure data: {e}")
        sys.exit(1)
else:
    reconstruction_errors_failure = np.array([])

# Combine reconstruction errors and true labels for ROC curve
all_reconstruction_errors = np.concatenate(
    [reconstruction_errors_normal, reconstruction_errors_failure]
)
all_true_labels = np.concatenate([y_test_normal_labels, y_test_failure_labels])

# Calculate threshold based on normal data
if len(reconstruction_errors_normal) > 0:
    threshold = np.percentile(reconstruction_errors_normal, 95)
    print(f"\nAnomaly Threshold (95th percentile of normal data): {threshold:.4f}")
else:
    threshold = 1.0
    print(f"\nWarning: Using default threshold: {threshold:.4f}")

# Calculate detection rates and save anomaly records
print("\n--- Detection Rate Analysis ---")
if len(reconstruction_errors_failure) > 0:
    # Calculate detection rate for failure data
    detected_failures = sum(
        1 for error in reconstruction_errors_failure if error > threshold
    )
    total_failures = len(reconstruction_errors_failure)
    detection_rate = (
        (detected_failures / total_failures) * 100 if total_failures > 0 else 0
    )

    # Calculate false positive rate for normal test data
    false_positives = sum(
        1 for error in reconstruction_errors_normal if error > threshold
    )
    total_normal = len(reconstruction_errors_normal)
    false_positive_rate = (
        (false_positives / total_normal) * 100 if total_normal > 0 else 0
    )

    print(f"Total failure sequences: {total_failures}")
    print(f"Detected failures: {detected_failures}")
    print(f"Detection rate: {detection_rate:.2f}%")
    print(f"Total normal test sequences: {total_normal}")
    print(f"False positives: {false_positives}")
    print(f"False positive rate: {false_positive_rate:.2f}%")

    # Create output directories
    output_dir = "./dataset/anomalies"
    plots_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Save detection rates
    detection_rates_path = os.path.join(output_dir, f"detection_rates_{table_name}.txt")
    with open(detection_rates_path, "w") as f:
        f.write(f"Detection Rate Analysis for {table_name}\n")
        f.write(f"Threshold: {threshold:.4f}\n")
        f.write(f"Total failure sequences: {total_failures}\n")
        f.write(f"Detected failures: {detected_failures}\n")
        f.write(f"Detection rate: {detection_rate:.2f}%\n")
        f.write(f"Total normal test sequences: {total_normal}\n")
        f.write(f"False positives: {false_positives}\n")
        f.write(f"False positive rate: {false_positive_rate:.2f}%\n")

    # Save anomaly records
    anomaly_records_path = os.path.join(output_dir, f"LSTM_AE_anomalies_{table_name}.csv")

    # Get indices of anomalies (both false positives and true failures)
    anomaly_indices_normal = np.where(reconstruction_errors_normal > threshold)[0]
    anomaly_indices_failure = np.where(reconstruction_errors_failure > threshold)[0]

    # Create DataFrame with anomaly records
    anomaly_records = []

    # Add failure anomalies only
    for idx in anomaly_indices_failure:
        original_idx = corresponding_failure_pdf_indices[idx]
        record = original_full_pdf.iloc[original_idx].to_dict()
        record["reconstruction_error"] = reconstruction_errors_failure[idx]
        record["is_failure"] = True
        anomaly_records.append(record)

    # Convert to DataFrame and save only the failure anomaly records
    anomaly_df = pd.DataFrame(anomaly_records)
    anomaly_df.to_csv(anomaly_records_path, index=False)
    print(f"\nFailure anomaly records saved to: {anomaly_records_path}")

    # Calculate and plot ROC curve
    if len(all_true_labels) > 0 and len(np.unique(all_true_labels)) > 1:
        try:
            fpr, tpr, roc_thresholds = roc_curve(
                all_true_labels, all_reconstruction_errors
            )
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(10, 6))
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (area = {roc_auc:.4f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Receiver Operating Characteristic (ROC) - {table_name}")
            plt.legend(loc="lower right")
            plt.grid(True)

            # Save ROC curve plot to plots directory
            roc_plot_path = os.path.join(plots_dir, f"ROC_curve_{table_name}.png")
            plt.savefig(roc_plot_path)
            plt.close()
            print(f"ROC curve saved to {roc_plot_path}")
        except Exception as e:
            print(f"Error calculating ROC curve: {e}")
    else:
        print("\nNot enough data or only one class available for ROC curve generation.")
else:
    print("No failure data available for detection rate analysis.")

print("\nScript finished.")
