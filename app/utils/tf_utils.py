import tensorflow as tf
import numpy as np  # Needed for calculate_percentile potentially via .numpy() if used outside TF graph
from keras import (
    layers,
)  # Needed for create_sequences_tf if layers used (not currently)


# --- TensorFlow Configuration ---
def configure_tensorflow():
    """Configures TensorFlow for performance and GPU memory growth."""
    # Enable mixed precision for better performance
    try:  # Add try-except for environments where mixed precision might not be supported/needed
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision policy set to 'mixed_float16'")
    except Exception as e:
        print(f"Could not set mixed precision policy: {e}")

    # Configure TensorFlow for better performance
    # tf.config.optimizer.set_jit(True) # JIT was causing issues, keep disabled for now
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
            "debug_stripper": True,
            "disable_model_pruning": False,
            "scoped_allocator_optimization": True,
            "pin_to_host_optimization": True,
            "implementation_selector": True,
            "auto_mixed_precision": False,  # Disabled for M2 compatibility
        }
    )

    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        print(f"Found {len(physical_devices)} physical GPU(s).")
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Enabled memory growth for GPU: {device.name}")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
    else:
        print("No physical GPUs detected.")

    # Configure for better performance
    # Setting threads might be better left to environment variables or higher-level config
    # tf.config.threading.set_inter_op_parallelism_threads(4)
    # tf.config.threading.set_intra_op_parallelism_threads(4)
    print("TensorFlow performance options configured.")


# --- Sequence Creation (TensorFlow) ---
def create_sequences_tf(data, timesteps):
    """
    Create sequences for LSTM using TensorFlow operations.

    Args:
        data: TensorFlow tensor of shape (n_samples, n_features)
        timesteps: Number of timesteps for each sequence

    Returns:
        TensorFlow tensor of shape (n_sequences, timesteps, n_features)
    """
    n_samples = tf.shape(data)[0]
    n_features = tf.shape(data)[1]

    # Ensure timesteps is a Tensor for TF operations
    timesteps_tf = tf.constant(timesteps, dtype=tf.int32)

    # Calculate number of sequences
    n_sequences = n_samples - timesteps_tf + 1

    # Validate sequence length dynamically
    if tf.less_equal(n_sequences, 0):
        print(
            f"Warning: Not enough samples ({n_samples.numpy()}) for timesteps ({timesteps}). Adjusting."
        )
        # Adjust timesteps to fit the data, ensuring at least 1 sequence
        timesteps_tf = tf.maximum(
            1, n_samples - 1
        )  # Adjusted timesteps must allow at least 1 sequence
        n_sequences = tf.maximum(
            1, n_samples - timesteps_tf + 1
        )  # Recalculate n_sequences, ensure it's at least 1
        print(
            f"Adjusted timesteps to {timesteps_tf.numpy()}, sequences to {n_sequences.numpy()}"
        )

    # Use tf.TensorArray for dynamic writing if shape issues persist, but tf.stack is often cleaner
    # sequences = tf.TensorArray(tf.float32, size=n_sequences)
    # for i in tf.range(n_sequences):
    #     sequences = sequences.write(i, data[i : i + timesteps_tf])
    # sequences_tensor = sequences.stack()

    # More direct approach using tf.stack and list comprehension (or tf.map_fn)
    indices = tf.range(n_sequences)
    sequences_tensor = tf.map_fn(
        lambda i: data[i : i + timesteps_tf], indices, dtype=tf.float32
    )

    # Final check and reshape (should typically match target shape)
    final_shape = [n_sequences, timesteps_tf, n_features]
    if not tf.reduce_all(tf.shape(sequences_tensor) == final_shape):
        try:
            sequences_tensor = tf.reshape(sequences_tensor, final_shape)
            print(f"Reshaped sequence tensor to: {final_shape}")
        except Exception as e:
            print(
                f"Error reshaping sequences: {e}. Shape was: {tf.shape(sequences_tensor).numpy()}, Target: {final_shape}"
            )
            raise

    return sequences_tensor


# --- Percentile Calculation ---
def calculate_percentile(data, percentile):
    """
    Calculate percentile using TensorFlow operations.

    Args:
        data: TensorFlow tensor (1D expected for percentile)
        percentile: Percentile value (0-100)

    Returns:
        TensorFlow tensor containing the percentile value
    """
    # Ensure data is float32 and 1D
    data = tf.cast(tf.reshape(data, [-1]), dtype=tf.float32)

    # Convert percentile to tensor and ensure float32 type
    percentile_tensor = tf.constant(percentile, dtype=tf.float32)

    # Sort the data
    sorted_data = tf.sort(data)

    # Calculate index for the percentile
    n_samples = tf.cast(tf.shape(sorted_data)[0], tf.float32)
    index = (n_samples - 1.0) * percentile_tensor / 100.0

    # Handle exact integer index and interpolation for float index
    index_floor = tf.floor(index)
    index_ceil = tf.math.ceil(index)

    # Values at floor and ceiling indices
    value_floor = sorted_data[tf.cast(index_floor, tf.int32)]
    value_ceil = sorted_data[tf.cast(index_ceil, tf.int32)]

    # Interpolate
    interpolated_value = value_floor + (index - index_floor) * (
        value_ceil - value_floor
    )

    # Use tf.cond to handle edge cases or exact matches cleanly
    # If index is integer, result is value_floor (or value_ceil as they are the same)
    result = tf.where(
        tf.equal(index_floor, index_ceil), value_floor, interpolated_value
    )

    return result
