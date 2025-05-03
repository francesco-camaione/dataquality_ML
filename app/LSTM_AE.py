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

table_name = "fraud_detection"
spark = (
    SparkSession.builder.appName("App")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.executor.memory", "5g")
    .getOrCreate()
)

df = (
    spark.read.option("header", "true")
    .csv(
        f"./dataset/{table_name}.csv",
        inferSchema=True,
    )
    .limit(10000)
)
categorical_cols, numerical_cols = infer_column_types_from_schema(df.schema)

# maps strings to numbers
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="skip")
    for c in categorical_cols
]
feature_cols = [c + "_idx" for c in categorical_cols] + numerical_cols

# combines the features values all into a single vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")

# standardize the assembled features values so that they have a mean of 0 and a standard deviation of 1
# to makes the ML model less sensitive to the scale of the features
scaler = StandardScaler(
    inputCol="assembled_features", outputCol="features", withMean=True, withStd=True
)

pipeline = Pipeline(stages=indexers + [assembler, scaler])
fitted_pipeline = pipeline.fit(df)
processed_df = fitted_pipeline.transform(df)

pdf = processed_df.select("features").toPandas()

# convert features to numpy array and finally the whole to a python array
feature_list = pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
# all values are stored using 32-bit floats
data = np.array(feature_list, dtype=np.float32)

# Ensure the data is reshaped correctly for LSTM
timesteps = 20
X = create_sequences(data, timesteps)

print(f"Data shape: {data.shape}")
print(f"Sequence shape: {X.shape}")

# Split into training and testing sets (e.g., 80% training)
train_size = int(0.8 * len(X))
X_train = X[:train_size]
X_test = X[train_size:]


n_features = X_train.shape[2]

# input_seq = Input(shape=(timesteps, n_features))

# # Encoder
# encoded = LSTM(32, activation='relu', return_sequences=True)(input_seq)
# encoded = LSTM(16, activation='relu')(encoded)

# # Latent space
# decoded = RepeatVector(timesteps)(encoded)

# # Decoder
# decoded = LSTM(16, activation='relu', return_sequences=True)(decoded)
# decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
# decoded = TimeDistributed(Dense(n_features))(decoded)

# # Build the model
# lstm_ae = Model(inputs=input_seq, outputs=decoded)
# lstm_ae.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
# lstm_ae.summary()

# # training
# history = lstm_ae.fit(
#     X_train, X_train,
#     epochs=11,
#     batch_size=32,
#     validation_split=0.2,
#     shuffle=True
# )

# X_test_pred = lstm_ae.predict(X_test)

# reconstruction_errors =  ms_error(X_test, X_test_pred)

# # Define a threshold for anomaly detection (e.g., the 95th percentile)
# threshold = np.percentile(reconstruction_errors, 95)
# print(f"Threshold : {threshold:.2f}")

# anomalies = reconstruction_errors > threshold

# lstm_ae.save(f"./ml_models/LSTM_AE_{table_name}.keras")
loaded_model = load_model(f"./ml_models/LSTM_AE_{table_name}.keras")
X_test_pred = loaded_model.predict(X_test)
reconstruction_errors = ms_error(X_test, X_test_pred)
threshold = np.percentile(reconstruction_errors, 95)
anomalies = reconstruction_errors > threshold

pdf["isFraud"] = df.select("isFraud").toPandas().values

# print(f"Detected {np.sum(anomalies)} anomalies in the test set out of {len(X_test)} sequences.")
# plot_reconstruction_error(
#     model_path=f"./ml_models/LSTM_AE_{table_name}.keras", test_data=X_test, percentile=95
# )
# Assuming 'pdf' is the DataFrame containing the original data
print("Details of the detected anomalies")
print("anomalies: ", anomalies)

for i in range(anomalies.shape[0]):  # Loop over each sequence
    if np.any(
        anomalies[i]
    ):  # Check if any feature in the current sequence is an anomaly
        error = reconstruction_errors[i]  # Get the corresponding reconstruction error
        print(
            f"Anomaly detected at sequence index {i} with reconstruction error: {error}"
        )
        # Print the corresponding original data record
        print(
            "Original data record:", pdf.iloc[train_size + i].to_dict()
        )  # Adjust index for test set # Adjust index for test set# Adjust index for test set

# Assuming 'pdf' is your original DataFrame with 10,000 rows
test_size = len(X_test)
pdf["predicted_anomaly"] = np.zeros(len(pdf))  # Initialize with zeros for all rows
print("+* COL: ", pdf.columns)

# Assign anomalies detected in the test set
for i in range(test_size):
    if i < len(anomalies):  # Ensure index is within anomalies length
        pdf.iloc[train_size + i, pdf.columns.get_loc("predicted_anomaly")] = int(
            np.any(anomalies[i])
        )


# Calculate True Positives, False Positives, True Negatives, False Negatives based on test set
TP = np.sum(
    (pdf["predicted_anomaly"][train_size:] == 1) & (pdf["isFraud"][train_size:] == 1)
)
FP = np.sum(
    (pdf["predicted_anomaly"][train_size:] == 1) & (pdf["isFraud"][train_size:] == 0)
)
TN = np.sum(
    (pdf["predicted_anomaly"][train_size:] == 0) & (pdf["isFraud"][train_size:] == 0)
)
FN = np.sum(
    (pdf["predicted_anomaly"][train_size:] == 0) & (pdf["isFraud"][train_size:] == 1)
)

# Calculate Precision, Recall, and F1 Score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = (
    2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
# Calculate Precision, Recall, and F1 Score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = (
    2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

# Optionally, visualize the reconstruction errors
plt.figure(figsize=(10, 6))
plt.hist(reconstruction_errors, bins=50, alpha=0.5, label="Reconstruction Errors")
plt.axvline(threshold, color="r", linestyle="dashed", linewidth=2, label="Threshold")
plt.title("Reconstruction Errors Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()
