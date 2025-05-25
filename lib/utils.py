import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType, StringType, BooleanType, NumericType


def boolean_columns(schema: StructType) -> list[str]:
    bool_cols = []
    for field in schema.fields:
        if isinstance(field.dataType, BooleanType):
            bool_cols.append(field.name)
    return bool_cols


def infer_column_types_from_schema(schema: StructType) -> tuple[list[str], list[str]]:
    """
    Infers categorical and numerical column names from a DataFrame schema.

    Parameters:
        schema (StructType): The schema of a Spark DataFrame.

    Returns:
        tuple: (categorical_columns, numerical_columns)
    """
    categorical_cols = []
    numerical_cols = []

    for field in schema.fields:
        if isinstance(field.dataType, StringType):
            categorical_cols.append(field.name)
        elif isinstance(field.dataType, (NumericType, BooleanType)):
            numerical_cols.append(field.name)

    return categorical_cols, numerical_cols


def ms_error_ae(test_data, reconstructed_data):
    # Calculate variance for each feature
    feature_variances = np.var(test_data, axis=0)
    # Avoid division by zero by replacing zero variances with 1
    feature_variances = np.where(feature_variances == 0, 1, feature_variances)
    # Calculate weighted MSE
    squared_errors = np.power(test_data - reconstructed_data, 2)
    weighted_errors = squared_errors / feature_variances
    return np.mean(weighted_errors, axis=1)


def mae_error_ae(test_data, reconstructed_data):
    # Calculate variance for each feature
    feature_variances = np.var(test_data, axis=0)
    # Set a minimum variance threshold to avoid numerical instability
    min_variance = 1e-6
    feature_variances = np.maximum(feature_variances, min_variance)
    # Calculate absolute errors
    abs_errors = np.abs(test_data - reconstructed_data)
    # Weight the errors by feature variances
    weighted_errors = abs_errors / feature_variances
    # Clip extreme values to avoid infinity
    weighted_errors = np.clip(weighted_errors, 0, 1e6)
    return np.mean(weighted_errors, axis=1)


def ms_error(test_data, reconstructed_data):
    # Calculate variance for each feature across all timesteps and samples
    feature_variances = np.var(test_data.reshape(-1, test_data.shape[-1]), axis=0)
    # Set a minimum variance threshold to avoid numerical instability
    min_variance = 1e-6
    feature_variances = np.maximum(feature_variances, min_variance)
    # Calculate squared errors
    squared_errors = np.power(test_data - reconstructed_data, 2)
    # Weight the errors by feature variances
    weighted_errors = squared_errors / feature_variances
    # Clip extreme values to avoid infinity
    weighted_errors = np.clip(weighted_errors, 0, 1e6)
    # Average across both timesteps and features
    return np.mean(weighted_errors, axis=(1, 2))


def mae_error(test_data, reconstructed_data):
    # Calculate variance for each feature across all timesteps and samples
    feature_variances = np.var(test_data.reshape(-1, test_data.shape[-1]), axis=0)
    # Avoid division by zero by replacing zero variances with 1
    feature_variances = np.where(feature_variances == 0, 1, feature_variances)
    # Calculate absolute errors
    abs_errors = np.abs(test_data - reconstructed_data)
    # Weight the errors by feature variances
    weighted_errors = abs_errors / feature_variances
    # Average across both timesteps and features
    return np.mean(weighted_errors, axis=(1, 2))


def create_sequences(data: np.ndarray, timesteps: int) -> np.ndarray:
    """
    Create sequences of data for LSTM training.
    Each sequence is of shape (timesteps, n_features).
    Returns a 3D array of shape (num_sequences, timesteps, n_features).
    """
    if len(data) < timesteps:
        # Return empty array with correct shape
        n_features = data.shape[1] if data.ndim == 2 else 1
        return np.empty((0, timesteps, n_features), dtype=np.float32)

    sequences = []
    for i in range(len(data) - timesteps + 1):
        seq = data[i : i + timesteps]
        sequences.append(seq)
    return np.array(sequences, dtype=np.float32)


def plot_roc_curve(all_true_labels_arg, all_reconstruction_errors_arg, table_name_arg):
    import os
    from sklearn.metrics import roc_curve, auc

    """
    Generates and saves the ROC curve plot.
    """
    if len(all_true_labels_arg) > 0 and len(np.unique(all_true_labels_arg)) > 1:
        fpr, tpr, _ = roc_curve(all_true_labels_arg, all_reconstruction_errors_arg)
        roc_auc = auc(fpr, tpr)
        plots_dir = "./plots"
        os.makedirs(plots_dir, exist_ok=True)

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
        plt.title(f"Receiver Operating Characteristic (ROC) - {table_name_arg}")
        plt.legend(loc="lower right")
        plt.grid(True)

        roc_plot_path = os.path.join(plots_dir, f"ROC_Curve_AE_{table_name_arg}.png")
        plt.savefig(roc_plot_path)
        plt.close()
        print(f"ROC curve saved to {roc_plot_path}")
    return
