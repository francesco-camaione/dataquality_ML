import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from pyspark.sql import SparkSession, functions, types
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer
import keras
import numpy as np
from lib.utils import ms_error_ae, infer_column_types_from_schema, boolean_columns

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

if not os.path.exists(f"./ml_models/AE_{table_name}.keras"):
    # train the model
    input_dim = X_train.shape[1]
    input_layer = keras.layers.Input(shape=(input_dim,))

    # Encoder with L2 regularization
    encoded = keras.layers.Dense(
        256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
    )(input_layer)
    encoded = keras.layers.BatchNormalization()(encoded)
    encoded = keras.layers.Dropout(0.3)(encoded)
    encoded = keras.layers.Dense(
        128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
    )(encoded)
    encoded = keras.layers.BatchNormalization()(encoded)
    encoded = keras.layers.Dropout(0.3)(encoded)
    latent_space = keras.layers.Dense(
        64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
    )(encoded)
    latent_space = keras.layers.BatchNormalization()(latent_space)

    # Decoder with L2 regularization
    decoded = keras.layers.Dense(
        128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
    )(latent_space)
    decoded = keras.layers.BatchNormalization()(decoded)
    decoded = keras.layers.Dropout(0.3)(decoded)
    decoded = keras.layers.Dense(
        256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
    )(decoded)
    decoded = keras.layers.BatchNormalization()(decoded)
    decoded = keras.layers.Dropout(0.3)(decoded)
    decoded = keras.layers.Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = keras.models.Model(inputs=input_layer, outputs=decoded)

    # Use a fixed initial learning rate with ReduceLROnPlateau
    initial_lr = 0.001
    optimizer = keras.optimizers.legacy.Adam(learning_rate=initial_lr)
    autoencoder.compile(optimizer=optimizer, loss="mean_squared_error")

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
        epochs=100,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
    )

    autoencoder.save(f"./ml_models/AE_{table_name}.keras")

#  Load trained model & predict
autoencoder = keras.models.load_model(f"./ml_models/AE_{table_name}.keras")
reconstructed = autoencoder.predict(X_test)
mse = ms_error_ae(X_test, reconstructed)

reconstructed_train = autoencoder.predict(X_train)
mse_train_normal = ms_error_ae(X_train, reconstructed_train)

threshold = np.percentile(mse_train_normal, 95)
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
anomaly_table["reconstruction_error"] = mse[anomaly_idxs]

print("Anomaly table (raw fraud records flagged):")
print(anomaly_table)
anomaly_table.to_csv(f"./dataset/anomalies/anomalies_{table_name}.csv")

# Model statistics and ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 1. Get reconstruction errors for the training (normal) data
reconstructed_train = autoencoder.predict(X_train)
mse_train_normal = ms_error_ae(X_train, reconstructed_train)

# mse is already calculated for X_test (fraud data) in the existing code
mse_test = mse

# 2. Create combined true labels and scores for ROC analysis
y_true = np.concatenate([np.zeros(len(mse_train_normal)), np.ones(len(mse_test))])

# Scores: reconstruction errors
y_scores = np.concatenate([mse_train_normal, mse_test])

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
predictions_on_normal_data = mse_train_normal > threshold

all_predictions = np.concatenate([predictions_on_normal_data, anomalies])

print("Confusion Matrix (Threshold: {:.4f}):".format(threshold))

cm = confusion_matrix(y_true, all_predictions)
print(cm)

spark.stop()
