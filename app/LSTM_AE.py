import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from keras.optimizers import Adam
from lib.utils import create_sequences, ms_error, plot_reconstruction_error

spark = SparkSession.builder.appName("DQ App").getOrCreate()

df = spark.read.option("header", "true").csv(
    "./unpivoted_data_10k/part-00000-5141f42c-90aa-4ee7-9672-5f5198bdc394-c000.csv",
    inferSchema=True,
)

# define categorical and numerical columns
categorical_cols = [
    "location",
    "kind",
    "host",
    "method",
    "statusCode",
    "endpoint",
]
numerical_cols = ["interval_start"]

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

feature_list = pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
data = np.array(feature_list, dtype=np.float32)  

# Ensure the data is reshaped correctly for LSTM
timesteps = 15
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

# lstm_ae.save("./ml_models/LSTM_AE.keras")


#print(f"Detected {np.sum(anomalies)} anomalies in the test set out of {len(X_test)} sequences.")
plot_reconstruction_error(
    model_path="./ml_models/LSTM_AE.keras", test_data=X_test, percentile=95
)
# print details of the detected anomalies
# for i, error in enumerate(reconstruction_errors):
#     if anomalies[i]:
#         print(f"Sequence {i} reconstruction error: {error:.4f} (Anomaly)")
