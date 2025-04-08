import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from lib.utils import ms_error, plot_reconstruction_error


spark = SparkSession.builder.appName("DQ App").getOrCreate()

df = spark.read.option("header", "true").csv(
    "./unpivoted_data_10k/part-00000-5141f42c-90aa-4ee7-9672-5f5198bdc394-c000.csv",
    inferSchema=True,
)

# Define categorical and numerical columns
categorical_cols = [
    "location",
    "kind",
    "host",
    "method",
    "statusCode",
    "endpoint",
]
numerical_cols = ["interval_start"]


# Create StringIndexers for categorical columns
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="skip")
    for c in categorical_cols
]
feature_cols = [c + "_idx" for c in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")


# Standardize the features
scaler = StandardScaler(
    inputCol="assembled_features", outputCol="features", withMean=True, withStd=True
)

# Build pipeline
pipeline = Pipeline(stages=indexers + [assembler, scaler])
fitted_pipeline = pipeline.fit(df)
processed_df = fitted_pipeline.transform(df)

pipeline_path = "./pipelines/AE_pipeline"
fitted_pipeline.write().overwrite().save(pipeline_path)


features_df = processed_df.select("features").toPandas()
features = np.array([row.toArray() for row in features_df["features"]])

X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)


# # Build the Autoencoder model
# input_dim = X_train.shape[1]

# input_layer = keras.layers.Input(shape=(input_dim,))

# # Encoder
# encoded = keras.layers.Dense(128, activation="relu")(input_layer)
# encoded = keras.layers.BatchNormalization()(encoded)
# encoded = keras.layers.Dropout(0.2)(encoded)

# encoded = keras.layers.Dense(64, activation="relu")(encoded)
# encoded = keras.layers.BatchNormalization()(encoded)
# encoded = keras.layers.Dropout(0.2)(encoded)

# latent_space = keras.layers.Dense(32, activation="relu")(encoded)
# latent_space = keras.layers.BatchNormalization()(latent_space)

# # Decoder
# decoded = keras.layers.Dense(64, activation="relu")(latent_space)
# decoded = keras.layers.BatchNormalization()(decoded)
# decoded = keras.layers.Dropout(0.2)(decoded)

# decoded = keras.layers.Dense(128, activation="relu")(decoded)
# decoded = keras.layers.BatchNormalization()(decoded)
# decoded = keras.layers.Dropout(0.2)(decoded)

# decoded = keras.layers.Dense(input_dim, activation="sigmoid")(decoded)

# autoencoder = keras.models.Model(inputs=input_layer, outputs=decoded)

# initial_learning_rate = 0.0001
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate, decay_steps=1000, decay_rate=0.9
# )
# optimizer = keras.optimizers.legacy.Adam(learning_rate=lr_schedule)

# autoencoder.compile(optimizer=optimizer, loss="mean_squared_error")

# early_stopping = keras.callbacks.EarlyStopping(
#     monitor="val_loss", patience=10, restore_best_weights=True
# )

# # Train the Autoencoder with a smaller batch size
# history = autoencoder.fit(
#     X_train,
#     X_train,
#     epochs=100,
#     batch_size=128,
#     validation_data=(X_test, X_test),
#     callbacks=[early_stopping],
# )


# # Evaluate
# loss = autoencoder.evaluate(X_test, X_test)
# print(f"Test Loss: {loss}")


# # Reconstruction error
# reconstructed = autoencoder.predict(X_test)
# mse = ms_error(X_test, reconstructed)


# # Threshold for anomaly detection
# threshold = np.percentile(mse, 95)
# anomalies = mse > threshold
# print(f"Number of anomalies detected: {np.sum(anomalies)}")

# autoencoder.save("./ml_models/AE.keras")

plot_reconstruction_error(
    model_path="./ml_models/AE.keras", test_data=X_test, percentile=95
)