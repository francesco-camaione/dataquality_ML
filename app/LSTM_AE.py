import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from keras.models import Model, load_model
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from keras.optimizers.legacy import Adam
from lib.utils import (
    create_sequences,
    ms_error,
    plot_reconstruction_error,
    infer_column_types_from_schema,
)
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

table_name = "fraud_detection_sample"

# Initialize Spark session
spark = (
    SparkSession.builder.appName("App")
    .master("local[*]")  # Use local mode with all available cores
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.executor.memory", "8g")
    .getOrCreate()
)

# Load full dataset
original_df = (
    spark.read.option("header", "true")
    .csv(f"./dataset/{table_name}.csv", inferSchema=True)
    .limit(20000)
)

# Separate fraud and normal records
fraud_records = original_df.where(original_df["isFraud"] == 1)
df = original_df.where(original_df["isFraud"] == 0)  # Training on normal data only

# Infer schema column types
categorical_cols, numerical_cols = infer_column_types_from_schema(df.schema)

# Feature pipeline
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="skip")
    for c in categorical_cols
]
feature_cols = [c + "_idx" for c in categorical_cols] + numerical_cols

assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")
scaler = StandardScaler(
    inputCol="assembled_features", outputCol="features", withMean=True, withStd=True
)

pipeline = Pipeline(stages=indexers + [assembler, scaler])
fitted_pipeline = pipeline.fit(df)

# Process normal (training) data
processed_df = fitted_pipeline.transform(df)
pdf = processed_df.select("features").toPandas()
feature_list = pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
data = np.array(feature_list, dtype=np.float32)

# Prepare sequences
timesteps = 20
X = create_sequences(data, timesteps)

# Train-test split (normal data only)
train_size = int(0.8 * len(X))
X_train = X[:train_size]
X_test_normal = X[train_size:]

# Process fraud records (for testing only)
fitted_pipeline2 = pipeline.fit(fraud_records)
processed_fraud_df = fitted_pipeline2.transform(fraud_records)
fraud_pdf = processed_fraud_df.select("features").toPandas()
fraud_features = fraud_pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
fraud_data = np.array(fraud_features, dtype=np.float32)
X_test = create_sequences(fraud_data, timesteps)

# create labels for the test dataset (all ones since it contains only isFraud = 1)
y_test = np.ones(len(X_test))  # All labels are 1 for fraud

# Define model
# n_features = X_train.shape[2]
# input_seq = Input(shape=(timesteps, n_features))

# # Encoder
# encoded = LSTM(32, activation="relu", return_sequences=True)(input_seq)
# encoded = LSTM(16, activation="relu")(encoded)

# # Latent space
# decoded = RepeatVector(timesteps)(encoded)

# # Decoder
# decoded = LSTM(16, activation="relu", return_sequences=True)(decoded)
# decoded = LSTM(32, activation="relu", return_sequences=True)(decoded)
# decoded = TimeDistributed(Dense(n_features))(decoded)

# # Build and compile model
# lstm_ae = Model(inputs=input_seq, outputs=decoded)
# lstm_ae.compile(optimizer=Adam(learning_rate=0.005), loss="mse")
# lstm_ae.summary()

# # Train
# history = lstm_ae.fit(
#     X_train, X_train, epochs=11, batch_size=32, validation_split=0.2, shuffle=True
# )

# # Save the trained model
# lstm_ae.save(f"./ml_models/LSTM_AE_{table_name}.keras")

# Predict and evaluate
loaded_model = load_model(f"./ml_models/LSTM_AE_{table_name}.keras")

# Filter out None values and make predictions
X_test_valid = X_test[~np.array([x is None for x in X_test])]
X_test_pred = loaded_model.predict(X_test_valid)

# calculate reconstruction errors
reconstruction_errors = ms_error(X_test_valid, X_test_pred)
threshold = np.percentile(reconstruction_errors, 95)
print(f"Anomaly threshold: {threshold:.2f}")

anomalies = reconstruction_errors > threshold

# Records detected as anomalies
anomaly_indices = np.where(anomalies)[0]  # Get indices of anomalies
anomaly_records = fraud_records.toPandas()  # Convert original DataFrame to Pandas
anomaly_table = anomaly_records.iloc[anomaly_indices]
print("Anomaly table: ", anomaly_table)
# anomaly_table.to_csv(f"./dataset/anomaly_records_{table_name}.csv")

# Summary of fraud detection
total_fraud = len(y_test)  # Total fraud records passed to the model
detected_fraud = np.sum(anomalies)  # Number of anomalies flagged by the model

print(f"Total fraud records tested: {total_fraud}")
print(f"Fraud records correctly flagged: {detected_fraud}")
print(f"Detection rate  %: {100 * detected_fraud / total_fraud:.2f}%")


# plot_reconstruction_error(
#     test_data=y_test, percentile=95, bins=50, figsize=(10, 6), model=loaded_model
# )
