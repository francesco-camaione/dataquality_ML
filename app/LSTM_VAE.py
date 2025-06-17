import sys
import os
import tensorflow as tf
import json
import gc

tf.config.set_visible_devices([], "GPU")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from keras.models import Model, load_model
from keras.layers import (
    Input,
    LSTM,
    GRU,
    RepeatVector,
    TimeDistributed,
    Dense,
    Dropout,
    BatchNormalization,
    Bidirectional,
    Lambda,
)
from keras.optimizers.legacy import Adam
from keras.regularizers import l2
from keras import backend as K
from lib.connector import SparkToAWS
from lib.utils import (
    create_sequences,
    mae_error,
    plot_roc_curve,
    plot_reconstruction_error,
)
from pyspark.sql.functions import col
from pyspark.sql.types import NumericType
from pyspark.sql.functions import isnan


def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def vae_loss(x, x_decoded_mean, z_mean, z_log_var):
    """
    VAE loss function that combines reconstruction loss and KL divergence.

    Parameters:
        x: Original input
        x_decoded_mean: Reconstructed output
        z_mean: Mean of the latent space
        z_log_var: Log variance of the latent space

    Returns:
        Combined loss value
    """
    reconstruction_loss = tf.keras.losses.mean_squared_error(x, x_decoded_mean)
    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    )
    return reconstruction_loss + kl_loss


@tf.keras.saving.register_keras_serializable()
class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.reconstruction_loss = tf.keras.losses.MeanSquaredError()

    def call(self, inputs):
        x, x_decoded_mean, z_mean, z_log_var = inputs
        reconstruction_loss = self.reconstruction_loss(x, x_decoded_mean)
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)
        return x_decoded_mean


def preprocess_data(spark, train_table_name_arg, test_table_name_arg, timesteps_arg):
    """
    Loads, preprocesses both training and test data.
    Fits preprocessing pipeline on training data and applies it to both datasets.
    """
    # Columns to select as per user request
    columns_to_select_initial = [
        "date",
        "serial_number",
        "model",
        "capacity_bytes",
        "failure",
        "datacenter",
        "cluster_id",
        "vault_id",
        "pod_id",
        "pod_slot_num",
        "is_legacy_format",
        "smart_1_normalized",
        "smart_1_raw",
        "smart_2_normalized",
        "smart_2_raw",
        "smart_3_normalized",
        "smart_3_raw",
        "smart_4_normalized",
        "smart_4_raw",
        "smart_5_normalized",
        "smart_5_raw",
        "smart_7_normalized",
        "smart_7_raw",
        "smart_8_normalized",
        "smart_8_raw",
        "smart_9_normalized",
        "smart_9_raw",
        "smart_10_normalized",
        "smart_10_raw",
        "smart_11_normalized",
        "smart_11_raw",
        "smart_12_normalized",
        "smart_12_raw",
        "smart_13_normalized",
        "smart_13_raw",
        "smart_15_normalized",
        "smart_15_raw",
        "smart_16_normalized",
        "smart_16_raw",
        "smart_17_normalized",
        "smart_17_raw",
        "smart_18_normalized",
        "smart_18_raw",
    ]
    # Load training data
    train_df = (
        spark.read.option("header", "true")
        .csv(f"./dataset/{train_table_name_arg}.csv", inferSchema=True)
        .select(*columns_to_select_initial)
        .limit(300000)
    )

    # Load test data
    test_df = (
        spark.read.option("header", "true")
        .csv(
            f"./dataset/{test_table_name_arg}.csv",
            inferSchema=True,
        )
        .select(*columns_to_select_initial)
    )

    # --- Define Spark ML Preprocessing Pipeline based on training data ---
    all_numeric_cols = [
        f.name
        for f in train_df.schema.fields
        if isinstance(f.dataType, NumericType) and f.name.lower() != "failure"
    ]

    # Filter out numeric columns that are entirely null or NaN
    valid_numeric_cols = []
    print("Validating numeric columns for Imputer...")
    for col_name in all_numeric_cols:
        non_null_count = (
            train_df.select(col_name)
            .filter(col(col_name).isNotNull() & ~isnan(col(col_name)))
            .count()
        )
        if non_null_count > 0:
            valid_numeric_cols.append(col_name)
        else:
            print(
                f"Column '{col_name}' is entirely null or NaN and will be excluded from imputation and features."
            )

    if not valid_numeric_cols:
        print(
            "Error: No valid numeric columns found for the pipeline. Please check your CSV or customize column selection."
        )
        return None

    print(f"Using valid numeric columns for pipeline: {valid_numeric_cols}")

    stages = []

    # Stage 1: Imputer for missing values
    imputed_numeric_cols = [col_name + "_imputed" for col_name in valid_numeric_cols]
    imputer = Imputer(
        inputCols=valid_numeric_cols,
        outputCols=imputed_numeric_cols,
        strategy="median",
    )
    stages.append(imputer)

    # Stage 2: VectorAssembler
    assembler = VectorAssembler(
        inputCols=imputed_numeric_cols,
        outputCol="features_unscaled",
        handleInvalid="skip",
    )
    stages.append(assembler)

    # Stage 3: StandardScaler
    scaler = StandardScaler(
        inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True
    )
    stages.append(scaler)

    pipeline = Pipeline(stages=stages)

    # Fit pipeline on training data
    print("Fitting preprocessing pipeline on training data...")
    pipeline_model = pipeline.fit(train_df)
    print("Preprocessing pipeline fitted.")

    # Save pipeline artifacts
    pipeline_save_path = (
        f"./pipelines/LSTM_VAE_pipeline/LSTM_VAE_pipeline_{train_table_name_arg}"
    )
    pipeline_model.write().overwrite().save(pipeline_save_path)
    print(f"Full preprocessing pipeline model saved to: {pipeline_save_path}")

    # Transform both training and test data
    print("Transforming training data...")
    processed_train_df = pipeline_model.transform(train_df)
    print("Transforming test data...")
    processed_test_df = pipeline_model.transform(test_df)

    # Convert to pandas and prepare sequences
    columns_to_select = ["features"]
    if "failure" in processed_train_df.columns:
        columns_to_select.append("failure")

    # Process training data
    train_pdf = processed_train_df.select(columns_to_select).toPandas()
    train_feature_list = (
        train_pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
    )
    train_data = np.array(train_feature_list, dtype=np.float32)
    X_train_sequences = create_sequences(train_data, timesteps_arg)

    # Process test data
    test_pdf = processed_test_df.select(columns_to_select).toPandas()
    test_feature_list = (
        test_pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
    )
    test_data = np.array(test_feature_list, dtype=np.float32)
    X_test_sequences = create_sequences(test_data, timesteps_arg)

    # Prepare labels if failure column exists
    y_train = None
    y_test = None
    if "failure" in train_pdf.columns:
        y_train = train_pdf["failure"].values
        y_test = test_pdf["failure"].values

    return (
        train_df.toPandas(),
        test_df.toPandas(),
        X_train_sequences,
        X_test_sequences,
        y_train,
        y_test,
    )


def load_or_create_model(model_path_arg, timesteps_arg, n_features_arg, X_train_arg):
    """
    Loads an existing model or creates, trains, and saves a new one.
    """
    model_dir = os.path.dirname(model_path_arg)
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(model_path_arg):
        print(f"Loading existing model from {model_path_arg}")
        vae_model = load_model(
            model_path_arg,
            compile=False,
            custom_objects={
                "sampling": sampling,
                "vae_loss": vae_loss,
                "VAELossLayer": VAELossLayer,
            },
            safe_mode=False,
        )
        optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
        vae_model.compile(optimizer=optimizer, loss=vae_loss)
        print("Model loaded and compiled successfully.")
    else:
        print("Defining and training a new VAE model...")

        # Encoder
        input_seq = Input(shape=(timesteps_arg, n_features_arg))

        # Add Gaussian noise to input for better generalization
        x = tf.keras.layers.GaussianNoise(0.01)(input_seq)

        # Encoder LSTM layers
        x = LSTM(
            128,
            activation="tanh",
            return_sequences=True,
            kernel_regularizer=l2(0.0001),
            recurrent_regularizer=l2(0.0001),
            recurrent_dropout=0.1,
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = LSTM(
            64,
            activation="tanh",
            return_sequences=True,
            kernel_regularizer=l2(0.0001),
            recurrent_regularizer=l2(0.0001),
            recurrent_dropout=0.1,
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        # Latent space (bottleneck)
        x = LSTM(
            32,
            activation="tanh",
            return_sequences=False,
            kernel_regularizer=l2(0.0001),
            recurrent_regularizer=l2(0.0001),
            recurrent_dropout=0.1,
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # VAE specific layers - mean and variance of the latent space
        z_mean = Dense(16, name="z_mean")(x)
        z_log_var = Dense(16, name="z_log_var")(x)

        # Use reparameterization trick to push the sampling out as input
        z = Lambda(sampling, output_shape=(16,), name="z")([z_mean, z_log_var])

        # Decoder
        x = RepeatVector(timesteps_arg)(z)

        x = LSTM(
            64,
            activation="tanh",
            return_sequences=True,
            kernel_regularizer=l2(0.0001),
            recurrent_regularizer=l2(0.0001),
            recurrent_dropout=0.1,
        )(x)
        x = BatchNormalization()(x)

        x = LSTM(
            128,
            activation="tanh",
            return_sequences=True,
            kernel_regularizer=l2(0.0001),
            recurrent_regularizer=l2(0.0001),
        )(x)
        x = BatchNormalization()(x)

        # Output layer
        decoded = TimeDistributed(
            Dense(n_features_arg, activation="linear", kernel_regularizer=l2(0.0001))
        )(x)

        # Create the VAE model with the custom loss layer
        vae_output = VAELossLayer()([input_seq, decoded, z_mean, z_log_var])
        vae_model = Model(inputs=input_seq, outputs=vae_output)

        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        vae_model.compile(
            optimizer=optimizer, loss=None
        )  # Loss is handled by the custom layer
        vae_model.summary()

        # Enhanced training configuration
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True, min_delta=0.0001
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=10, min_lr=0.000001, verbose=1
        )

        # Train the model with optimized batch size
        history = vae_model.fit(
            X_train_arg,
            X_train_arg,
            epochs=100,
            batch_size=256,
            validation_split=0.2,
            shuffle=True,
            callbacks=[early_stopping, reduce_lr],
        )

        # Save the model after training
        try:
            vae_model.save(model_path_arg, save_format="keras")
            print(f"New model saved to {model_path_arg}")
        except Exception as e:
            print(f"Warning: Could not save model in keras format: {e}")
            try:
                vae_model.save(model_path_arg, save_format="tf")
                print(f"New model saved to {model_path_arg} in TensorFlow format")
            except Exception as e:
                print(f"Error saving model: {e}")
                print("Continuing with the trained model in memory...")

    return vae_model


def predict_and_evaluate(
    model, X_test_normal_arg, X_test_failure_arg, batch_size_arg=64
):
    """
    Predicts and evaluates the model on test data, using batching to conserve memory.
    """
    # Predict on normal test data
    print(f"Predicting on normal test data with batch size {batch_size_arg}...")
    reconstructed_normal = model.predict(X_test_normal_arg, batch_size=batch_size_arg)
    reconstruction_errors_normal = mae_error(X_test_normal_arg, reconstructed_normal)

    # Predict on failure test data
    print(f"Predicting on failure test data with batch size {batch_size_arg}...")
    reconstructed_failure = model.predict(X_test_failure_arg, batch_size=batch_size_arg)
    reconstruction_errors_failure = mae_error(X_test_failure_arg, reconstructed_failure)

    return reconstruction_errors_normal, reconstruction_errors_failure


def generate_anomaly_report(
    reconstruction_errors_normal_arg,
    reconstruction_errors_failure_arg,
    threshold_arg,
    original_full_pdf_arg,
    all_failure_original_indices_arg,
    timesteps_arg,
    table_name_arg,
):
    """
    Generates and saves the anomaly report, listing all unique original records
    from detected anomalous sequences.
    """
    anomaly_records = []
    added_original_indices = set()

    detected_failure_sequences_count = 0
    total_failure_sequences = len(reconstruction_errors_failure_arg)

    if total_failure_sequences > 0:
        anomalous_sequence_indices = np.where(
            reconstruction_errors_failure_arg > threshold_arg
        )[0]
        detected_failure_sequences_count = len(anomalous_sequence_indices)

        for sequence_idx in anomalous_sequence_indices:
            # This sequence (sequence_idx) is anomalous.
            # It was formed from failure_data[sequence_idx] to failure_data[sequence_idx + timesteps_arg - 1]
            for j in range(timesteps_arg):
                failure_record_inner_idx = sequence_idx + j
                if failure_record_inner_idx < len(all_failure_original_indices_arg):
                    original_pdf_index = all_failure_original_indices_arg[
                        failure_record_inner_idx
                    ]

                    if original_pdf_index not in added_original_indices:
                        record = original_full_pdf_arg.iloc[
                            original_pdf_index
                        ].to_dict()
                        # Assign the reconstruction error of the sequence to each record from it
                        record["reconstruction_error_sequence"] = (
                            reconstruction_errors_failure_arg[sequence_idx]
                        )
                        record["is_failure"] = (
                            True  # These records are from failure_data
                        )
                        anomaly_records.append(record)
                        added_original_indices.add(original_pdf_index)

    # Calculate false positive rate for normal test data
    false_positives_sequences = sum(
        1 for error in reconstruction_errors_normal_arg if error > threshold_arg
    )
    total_normal_sequences = len(reconstruction_errors_normal_arg)
    false_positive_rate = (
        (false_positives_sequences / total_normal_sequences * 100)
        if total_normal_sequences > 0
        else 0
    )

    detection_rate_sequences = (
        (detected_failure_sequences_count / total_failure_sequences * 100)
        if total_failure_sequences > 0
        else 0
    )

    print(f"Total failure sequences presented to model: {total_failure_sequences}")
    print(f"Detected anomalous failure sequences: {detected_failure_sequences_count}")
    print(f"Sequence-based detection rate: {detection_rate_sequences:.2f}%")
    print(
        f"Total unique original failure records identified in anomalous sequences: {len(added_original_indices)}"
    )
    print(f"Total normal test sequences: {total_normal_sequences}")
    print(
        f"False positive sequences (normal sequences flagged as anomaly): {false_positives_sequences}"
    )
    print(f"False positive rate (sequences): {false_positive_rate:.2f}%")

    # Save anomaly records
    if anomaly_records:
        output_dir = "./dataset/anomalies"
        os.makedirs(output_dir, exist_ok=True)
        anomaly_records_path = os.path.join(
            output_dir, f"LSTM_VAE_anomalies_{table_name_arg}.csv"
        )
        anomaly_df = pd.DataFrame(anomaly_records)
        print(f"Anomaly records: {anomaly_df}")
        anomaly_df.to_csv(anomaly_records_path, index=False)
        print(
            f"Unique original failure records from anomalous sequences saved to: {anomaly_records_path}"
        )
    else:
        print("No unique original failure records identified in anomalous sequences.")


def main():
    train_table_name = "2024_12_bb_3d"
    test_table_name = "2024_12_25_test"
    timesteps_main = 20
    connector = SparkToAWS()
    spark_main = connector.create_local_spark_session()

    (
        original_train_pdf,
        original_test_pdf,
        X_train_main,
        X_test_main,
        y_train_main,
        y_test_main,
    ) = preprocess_data(spark_main, train_table_name, test_table_name, timesteps_main)

    n_features_main = X_train_main.shape[2]
    model_path_main = os.path.join("./ml_models", f"LSTM_VAE_{train_table_name}.keras")

    # Check if model was loaded or newly trained
    model_was_loaded = os.path.exists(model_path_main)
    vae_model_main = load_or_create_model(
        model_path_main, timesteps_main, n_features_main, X_train_main
    )

    thresholds_dir = "./dataset/thresholds"
    threshold_file = os.path.join(
        thresholds_dir, f"LSTM_VAE_threshold_{train_table_name}.txt"
    )

    if model_was_loaded and os.path.exists(threshold_file):
        print(f"Loading existing threshold from {threshold_file}")
        with open(threshold_file, "r") as f:
            threshold_main = float(f.read())
        print(f"Threshold loaded: {threshold_main:.4f}")

        del X_train_main
        gc.collect()
        print("Training data cleared.")

    else:
        # Calculate threshold on training data reconstruction errors for a more robust baseline
        print("Calculating threshold from training data reconstruction errors...")
        reconstructed_train = vae_model_main.predict(X_train_main, batch_size=64)
        reconstruction_errors_train = mae_error(X_train_main, reconstructed_train)

        # Calculate dynamic threshold using IQR method
        threshold_main = np.percentile(
            reconstruction_errors_train, 95
        )  # Standard outlier detection threshold

        # Save threshold
        os.makedirs(thresholds_dir, exist_ok=True)
        with open(threshold_file, "w") as f:
            f.write(f"{threshold_main}")
        print(f"Threshold saved to: {threshold_file}")

        del X_train_main, reconstructed_train, reconstruction_errors_train
        gc.collect()
        print("Training data cleared.")

    # Predict on test data
    print("Predicting on test data...")
    reconstructed_test = vae_model_main.predict(X_test_main, batch_size=64)
    reconstruction_errors = mae_error(X_test_main, reconstructed_test)

    # --- Reconstruction Error Statistics for Test Dataset ---
    print("\n--- Reconstruction Error Statistics for Test Dataset ---")
    print(f"Min Error: {np.min(reconstruction_errors):.4f}")
    print(f"Max Error: {np.max(reconstruction_errors):.4f}")
    print(f"Mean Error: {np.mean(reconstruction_errors):.4f}")
    print(f"Median Error: {np.median(reconstruction_errors):.4f}")
    print(f"Std Dev of Error: {np.std(reconstruction_errors):.4f}")
    print("--------------------------------------------------\n")

    # For plotting and evaluation, we need to align errors with original data
    padding_size = len(original_test_pdf) - len(reconstruction_errors)

    # Check if true labels are available for rich plotting
    true_labels_for_plot = None
    if y_test_main is not None:
        true_labels_for_plot = y_test_main[padding_size:]

    # Plot reconstruction error
    plot_reconstruction_error(
        reconstruction_errors,
        threshold_main,
        test_table_name,
        true_labels=true_labels_for_plot,
    )

    # Generate anomaly predictions
    anomaly_predictions = reconstruction_errors > threshold_main

    # Save anomalies
    output_dir = "./dataset/anomalies"
    os.makedirs(output_dir, exist_ok=True)
    anomalies_path = os.path.join(
        output_dir, f"LSTM_VAE_anomalies_{test_table_name}.csv"
    )

    # Create results DataFrame to work with
    results_df = original_test_pdf.copy()

    results_df["reconstruction_error"] = np.pad(
        reconstruction_errors, (padding_size, 0), constant_values=np.nan
    )
    results_df["is_anomaly"] = np.pad(
        anomaly_predictions, (padding_size, 0), constant_values=False
    )
    if y_test_main is not None:
        results_df["actual_failure"] = y_test_main

    # Filter for anomalies and save them
    anomalies_df = results_df[results_df["is_anomaly"] == True].copy()
    anomalies_df.to_csv(anomalies_path, index=False)
    print(f"Found {len(anomalies_df)} anomalies. Saved to: {anomalies_path}")

    if y_test_main is not None:
        # --- Performance Statistics ---
        # We only consider records for which a prediction was possible (i.e., after the initial sequence window)
        eval_df = results_df.iloc[padding_size:].copy()
        eval_df["actual_failure"] = eval_df["actual_failure"].astype(bool)
        eval_df["is_anomaly"] = eval_df["is_anomaly"].astype(bool)

        true_positives = len(
            eval_df[
                (eval_df["is_anomaly"] == True) & (eval_df["actual_failure"] == True)
            ]
        )
        false_positives = len(
            eval_df[
                (eval_df["is_anomaly"] == True) & (eval_df["actual_failure"] == False)
            ]
        )

        total_actual_failures = eval_df["actual_failure"].sum()
        detection_rate = (
            (true_positives / total_actual_failures * 100)
            if total_actual_failures > 0
            else 0
        )

        print("\n--- Model Performance Statistics ---")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"Total actual failures in evaluated data: {total_actual_failures}")
        print(f"Detection Rate: {detection_rate:.2f}%")
        print("------------------------------------\n")

        plot_roc_curve(
            y_test_main[padding_size:], reconstruction_errors, test_table_name
        )

    connector.close_spark_session()
    print("Script finished.")


if __name__ == "__main__":
    main()
