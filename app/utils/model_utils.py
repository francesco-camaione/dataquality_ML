import tensorflow as tf
from keras import layers


def create_model(n_features, timesteps):
    """Creates the LSTM Autoencoder Keras model."""
    input_seq = layers.Input(shape=(timesteps, n_features))

    # Encoder
    encoded = layers.LSTM(32, activation="tanh", return_sequences=True)(input_seq)
    encoded = layers.LSTM(16, activation="tanh")(encoded)

    # Latent space
    decoded = layers.RepeatVector(timesteps)(encoded)

    # Decoder
    decoded = layers.LSTM(16, activation="tanh", return_sequences=True)(decoded)
    decoded = layers.LSTM(32, activation="tanh", return_sequences=True)(decoded)
    decoded = layers.TimeDistributed(layers.Dense(n_features, activation="linear"))(
        decoded
    )

    model = tf.keras.Model(inputs=input_seq, outputs=decoded)

    # Use legacy optimizer for M2 compatibility / or non-legacy if appropriate
    try:
        # Try non-legacy first
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    except AttributeError:
        print("Using legacy Adam optimizer.")
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    # Apply LossScaleOptimizer if mixed precision is enabled
    # Check the global policy
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        print("Wrapping optimizer with LossScaleOptimizer for mixed precision.")
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    else:
        print(
            "Mixed precision not enabled or not using 'mixed_float16', using standard optimizer."
        )

    model.compile(
        optimizer=optimizer, loss="mse", jit_compile=False
    )  # Keep XLA disabled
    print("LSTM AE model created and compiled.")
    model.summary()  # Print model summary
    return model
