import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from pyspark.sql import SparkSession, functions, types
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer
import keras
import numpy as np
from lib.utils import mae_error_ae, infer_column_types_from_schema, boolean_columns

table_name = "2024_12_bb_3d"

spark = (
    SparkSession.builder.appName("App")
    .master("local[*]")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

# Load and split dataset
raw_df = (
    spark.read.option("header", "true").csv(
        f"./dataset/{table_name}.csv", inferSchema=True
    )
).repartition(32)

print("df has ", raw_df.count(), " records")

boolean_cols = boolean_columns(raw_df.schema)
if boolean_cols:
    for column in boolean_cols:
        raw_df = raw_df.withColumn(
            column, functions.col(column).cast(types.IntegerType())
        )

# Separate normal vs fraud without caching
df_norm = raw_df.where(raw_df["failure"] == 0)
df_fraud = raw_df.where(raw_df["failure"] == 1)
print("number of failure records in the raw_df: ", df_fraud.count())
raw_df.unpersist()

# Infer schema & build feature pipeline
categorical_cols, numerical_cols = infer_column_types_from_schema(df_norm.schema)

# Identify numerical columns that are entirely null or NaN in df_norm
problematic_numerical_cols = []
for col_name in numerical_cols:
    if (
        df_norm.where(
            functions.col(col_name).isNotNull()
            & ~functions.isnan(functions.col(col_name))
        ).count()
        == 0
    ):
        problematic_numerical_cols.append(col_name)
        print(
            f"Warning: Numerical column '{col_name}' contains only null/NaN values in df_norm and will be excluded from imputation and features."
        )

valid_numerical_cols = [
    c for c in numerical_cols if c not in problematic_numerical_cols
]

# Process data in a single pipeline to minimize memory usage
print("Building and fitting pipeline...")
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in categorical_cols
]
imputer = Imputer(
    inputCols=valid_numerical_cols, outputCols=valid_numerical_cols, strategy="mean"
)
feature_cols = (
    [c + "_idx" for c in categorical_cols] + valid_numerical_cols + boolean_cols
)
assembler = VectorAssembler(
    inputCols=feature_cols, outputCol="assembled_features", handleInvalid="keep"
)
scaler = StandardScaler(
    inputCol="assembled_features", outputCol="features", withMean=True, withStd=True
)

pipeline = Pipeline(stages=[imputer] + indexers + [assembler, scaler])
fitted_pipeline = pipeline.fit(df_norm)

print("Transforming normal data...")
proc_norm_df = fitted_pipeline.transform(df_norm)

print("Transforming fraud data...")
proc_fraud_df = fitted_pipeline.transform(df_fraud)

# Convert to numpy arrays in batches to minimize memory usage
print("Converting normal data to numpy arrays...")
norm_vectors = []
for row in proc_norm_df.select("features").collect():
    norm_vectors.append(row.features.toArray())
X_train = np.vstack(norm_vectors).astype(np.float32)

print("Converting fraud data to numpy arrays...")
fraud_vectors = []
for row in proc_fraud_df.select("features").collect():
    fraud_vectors.append(row.features.toArray())
X_test = np.vstack(fraud_vectors).astype(np.float32)

# Clean up DataFrames
df_norm.unpersist()
df_fraud.unpersist()
proc_norm_df.unpersist()
proc_fraud_df.unpersist()

model_path = f"./ml_models/AE_{table_name}.keras"
thresholds_dir = "./dataset/thresholds"
threshold_file_path = os.path.join(thresholds_dir, f"AE_threshold_{table_name}.txt")
model_was_loaded = os.path.exists(model_path)

if not model_was_loaded:
    # train the model
    input_dim = X_train.shape[1]
    input_layer = keras.layers.Input(shape=(input_dim,))

    # Encoder with L2 regularization
    encoded = keras.layers.Dense(
        512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
    )(input_layer)
    encoded = keras.layers.BatchNormalization()(encoded)
    encoded = keras.layers.Dense(
        256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
    )(encoded)
    encoded = keras.layers.BatchNormalization()(encoded)
    encoded = keras.layers.Dense(
        128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
    )(encoded)
    encoded = keras.layers.BatchNormalization()(encoded)
    latent_space = keras.layers.Dense(
        64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
    )(encoded)
    latent_space = keras.layers.BatchNormalization()(latent_space)

    # Decoder with L2 regularization
    decoded = keras.layers.Dense(
        128, activation="linear", kernel_regularizer=keras.regularizers.l2(0.001)
    )(latent_space)
    decoded = keras.layers.BatchNormalization()(decoded)
    decoded = keras.layers.Dense(
        256, activation="linear", kernel_regularizer=keras.regularizers.l2(0.001)
    )(decoded)
    decoded = keras.layers.BatchNormalization()(decoded)
    decoded = keras.layers.Dense(
        512, activation="linear", kernel_regularizer=keras.regularizers.l2(0.001)
    )(decoded)
    decoded = keras.layers.BatchNormalization()(decoded)
    decoded = keras.layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = keras.models.Model(inputs=input_layer, outputs=decoded)

    # Use a fixed initial learning rate with ReduceLROnPlateau
    initial_lr = 0.001
    optimizer = keras.optimizers.legacy.Adam(learning_rate=initial_lr)
    autoencoder.compile(optimizer=optimizer, loss="mae")

    # Enhanced early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, min_delta=0.0001
    )

    # Learning rate reduction on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
    )

    history = autoencoder.fit(
        X_train,
        X_train,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
    )

    autoencoder.save(f"./ml_models/AE_{table_name}.keras")

#  Load trained model & predict
autoencoder = keras.models.load_model(
    f"./ml_models/AE_{table_name}.keras", compile=False
)
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)

autoencoder.compile(optimizer=optimizer, loss="mae")
reconstructed = autoencoder.predict(X_test)
mae = mae_error_ae(X_test, reconstructed)

reconstructed_train = autoencoder.predict(X_train)
mae_train_normal = mae_error_ae(X_train, reconstructed_train)

if not model_was_loaded:
    threshold = np.percentile(mae_train_normal, 95)
    with open(threshold_file_path, "w") as f:
        f.write(str(threshold))
    print(f"Threshold calculated for new model and saved to: {threshold_file_path}")
else:
    with open(threshold_file_path, "r") as f:
        threshold = float(f.read().strip())
        print(f"Threshold loaded from: {threshold_file_path}")

anomalies = mae > threshold

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

# Handle timestamp columns before converting to Pandas
df_fraud_for_pandas = df_fraud
for col_name, dtype in df_fraud.dtypes:
    if (
        dtype == "timestamp" or dtype == "timestamp_ntz"
    ):  # Handling both common timestamp types
        df_fraud_for_pandas = df_fraud_for_pandas.withColumn(
            col_name, functions.col(col_name).cast("string")
        )
raw_fraud_pdf = df_fraud_for_pandas.toPandas()

anomaly_table = raw_fraud_pdf.iloc[anomaly_idxs]
# Add reconstruction error to the anomaly table
anomaly_table["reconstruction_error"] = mae[anomaly_idxs]

print("Anomaly table (raw fraud records flagged):")
print(anomaly_table)
anomaly_table.to_csv(f"./dataset/anomalies/anomalies_{table_name}.csv")

# Model statistics and ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

# 1. Get reconstruction errors for the training (normal) data
reconstructed_train = autoencoder.predict(X_train)
mae_train_normal = mae_error_ae(X_train, reconstructed_train)

# mae is already calculated for X_test (fraud data) in the existing code
mae_test = mae

# 2. Create combined true labels and scores for ROC analysis
y_true = np.concatenate([np.zeros(len(mae_train_normal)), np.ones(len(mae_test))])

# Scores: reconstruction errors
y_scores = np.concatenate([mae_train_normal, mae_test])

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

# 5. Calculate and print Confusion Matrix and Classification Report
# Using the threshold based on normal data distribution
# Predictions for normal data using this threshold
predictions_on_normal_data = mae_train_normal > threshold

all_predictions = np.concatenate([predictions_on_normal_data, anomalies])

print("Confusion Matrix (Threshold: {:.4f}):".format(threshold))

cm = confusion_matrix(y_true, all_predictions)
print(cm)


# --- Function to test model on a new dataset ---
def test_model_on_new_dataset(
    spark_session,
    pipeline_model,
    trained_ae_model,
    anomaly_threshold,
    test_table_name_str,
):
    print(f"\n\n--- Testing with new dataset: {test_table_name_str}.csv ---")

    new_test_raw_df = (
        spark_session.read.option("header", "true")
        .csv(f"./dataset/{test_table_name_str}.csv", inferSchema=True)
        .repartition(32)
    )
    print(
        f"New test df ({test_table_name_str}.csv) has {new_test_raw_df.count()} records"
    )

    if "failure" not in new_test_raw_df.columns:
        print(
            f"Error: The new test dataset {test_table_name_str}.csv must contain a 'failure' column for evaluation. Skipping."
        )
        return

    # Accessing boolean_columns, functions, types from the global scope of the script
    new_boolean_cols = boolean_columns(new_test_raw_df.schema)
    if new_boolean_cols:
        for column in new_boolean_cols:
            new_test_raw_df = new_test_raw_df.withColumn(
                column, functions.col(column).cast(types.IntegerType())
            )

    print(
        f"Transforming new test data ({test_table_name_str}.csv) using the existing fitted pipeline..."
    )
    proc_new_test_df = pipeline_model.transform(new_test_raw_df)

    print(f"Converting new test data ({test_table_name_str}.csv) to numpy arrays...")
    collected_rows_for_numpy = proc_new_test_df.select("features", "failure").collect()

    if not collected_rows_for_numpy:
        print(
            f"Warning: No data in proc_new_test_df for {test_table_name_str}.csv after transformation. Skipping evaluation."
        )
        if "new_test_raw_df" in locals() and new_test_raw_df is not None:
            new_test_raw_df.unpersist()
        if "proc_new_test_df" in locals() and proc_new_test_df is not None:
            proc_new_test_df.unpersist()
        return

    # Accessing np from the global scope
    new_test_features_list = [
        row.features.toArray() for row in collected_rows_for_numpy
    ]
    X_new_test = np.array(new_test_features_list).astype(np.float32)
    y_true_new = np.array([row.failure for row in collected_rows_for_numpy]).astype(int)

    if X_new_test.ndim == 1:
        X_new_test = X_new_test.reshape(1, -1)

    if len(X_new_test) == 0:
        print(
            f"Warning: X_new_test is empty for {test_table_name_str}.csv. Skipping further evaluation."
        )
        if "new_test_raw_df" in locals() and new_test_raw_df is not None:
            new_test_raw_df.unpersist()
        if "proc_new_test_df" in locals() and proc_new_test_df is not None:
            proc_new_test_df.unpersist()
        return

    print(f"Predicting on new test data ({test_table_name_str}.csv)...")
    # Accessing mae_error_ae from the global scope
    reconstructed_new_test = trained_ae_model.predict(X_new_test)
    mae_new_test = mae_error_ae(X_new_test, reconstructed_new_test)

    predictions_new_test = mae_new_test > anomaly_threshold

    print(f"\n--- Evaluation Results for {test_table_name_str}.csv ---")
    print(f"Using Anomaly threshold: {anomaly_threshold:.4f}")

    actual_positives_new = np.sum(y_true_new)
    actual_negatives_new = len(y_true_new) - actual_positives_new
    print(f"Actual normal records in {test_table_name_str}.csv: {actual_negatives_new}")
    print(
        f"Actual failure records in {test_table_name_str}.csv: {actual_positives_new}"
    )

    predicted_anomalies_new = np.sum(predictions_new_test)
    print(
        f"Total records flagged as anomalies by model in {test_table_name_str}.csv: {predicted_anomalies_new}"
    )

    # Accessing confusion_matrix, accuracy_score from sklearn.metrics (imported globally)
    cm_new = confusion_matrix(y_true_new, predictions_new_test, labels=[0, 1])

    # Ensure cm_new.ravel() provides 4 values, otherwise handle potential error for datasets with only one class
    if cm_new.size == 4:
        tn_new, fp_new, fn_new, tp_new = cm_new.ravel()
    elif (
        cm_new.size == 1 and len(y_true_new) > 0
    ):  # Only one class predicted and present
        if y_true_new[0] == 0 and predictions_new_test[0] == 0:  # All are TN
            tn_new, fp_new, fn_new, tp_new = cm_new.item(), 0, 0, 0
        elif y_true_new[0] == 1 and predictions_new_test[0] == 1:  # All are TP
            tn_new, fp_new, fn_new, tp_new = 0, 0, 0, cm_new.item()
        elif y_true_new[0] == 0 and predictions_new_test[0] == 1:  # All are FP
            tn_new, fp_new, fn_new, tp_new = 0, cm_new.item(), 0, 0
        elif y_true_new[0] == 1 and predictions_new_test[0] == 0:  # All are FN
            tn_new, fp_new, fn_new, tp_new = 0, 0, cm_new.item(), 0
        else:  # Should not happen
            tn_new, fp_new, fn_new, tp_new = 0, 0, 0, 0
    else:  # Default to zeros if confusion matrix is not 2x2 (e.g. no samples in a class)
        print(
            "Warning: Confusion matrix is not 2x2. Metrics might be affected if classes are missing."
        )
        tn_new, fp_new, fn_new, tp_new = 0, 0, 0, 0  # default values

    print("\nConfusion Matrix (New Test Set):")
    print(f"Predicted:    Normal(0) Failure(1)")
    print(f"Actual Normal(0): {tn_new:6d} {fp_new:8d}")
    print(f"Actual Failure(1):{fn_new:6d} {tp_new:8d}")

    print(f"\nDetailed Metrics (New Test Set - {test_table_name_str}.csv):")
    print(f"True Positives (Failures correctly flagged as Failure): {tp_new}")
    print(f"False Positives (Normals incorrectly flagged as Failure): {fp_new}")
    print(f"True Negatives (Normals correctly flagged as Normal): {tn_new}")
    print(f"False Negatives (Failures incorrectly flagged as Normal): {fn_new}")

    accuracy_new = accuracy_score(y_true_new, predictions_new_test)
    print(f"\nAccuracy: {accuracy_new:.4f}")

    precision_failure = tp_new / (tp_new + fp_new) if (tp_new + fp_new) > 0 else 0
    recall_failure = tp_new / (tp_new + fn_new) if (tp_new + fn_new) > 0 else 0
    f1_failure = (
        2 * (precision_failure * recall_failure) / (precision_failure + recall_failure)
        if (precision_failure + recall_failure) > 0
        else 0
    )

    print(f"\nMetrics for 'Failure' class (1) in {test_table_name_str}.csv:")
    print(f"  Precision: {precision_failure:.4f} (TP / (TP + FP))")
    print(f"  Recall (Sensitivity): {recall_failure:.4f} (TP / (TP + FN))")
    print(f"  F1-score: {f1_failure:.4f}")

    specificity_normal = tn_new / (tn_new + fp_new) if (tn_new + fp_new) > 0 else 0
    print(f"\nMetrics for 'Normal' class (0) in {test_table_name_str}.csv:")
    print(f"  Specificity: {specificity_normal:.4f} (TN / (TN + FP))")


# --- Test with 2024_12_15_bb_3d dataset ---
new_test_table_name_param = "2024_12_15_test"
test_model_on_new_dataset(
    spark_session=spark,
    pipeline_model=fitted_pipeline,
    trained_ae_model=autoencoder,
    anomaly_threshold=threshold,
    test_table_name_str=new_test_table_name_param,
)


spark.stop()
