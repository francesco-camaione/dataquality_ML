import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType, StringType, BooleanType, NumericType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer
from pyspark.sql.functions import col, isnan


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


def mae_error_ae(test_data, reconstructed_data):
    """
    Calculate normalized Mean Absolute Error (MAE) between original and reconstructed data.
    Errors are normalized by feature variance to prevent features with higher variability
    from dominating the error calculation.

    Parameters:
        test_data: Original data array
        reconstructed_data: Reconstructed data array

    Returns:
        Per-record reconstruction errors, normalized by feature variance
    """
    # Calculate variance for each feature
    feature_variances = np.var(test_data, axis=0)

    # Set a minimum variance threshold to avoid numerical instability
    min_variance = 1e-6
    feature_variances = np.maximum(feature_variances, min_variance)

    # Calculate absolute errors for each feature
    abs_errors = np.abs(test_data - reconstructed_data)

    # Normalize errors by feature variance
    normalized_errors = abs_errors / feature_variances

    # Calculate per-record mean normalized error
    return np.mean(normalized_errors, axis=1)


def mae_error(test_data, reconstructed_data):
    """
    Calculate Mean Absolute Error (MAE) between original and reconstructed sequences.
    Errors are normalized by feature variance to prevent features with higher variability
    from dominating the error calculation.

    Parameters:
        test_data: Original data array of shape (n_sequences, timesteps, n_features)
        reconstructed_data: Reconstructed data array of same shape

    Returns:
        Per-sequence reconstruction errors
    """
    # Calculate variance for each feature across all timesteps and samples
    feature_variances = np.var(test_data.reshape(-1, test_data.shape[-1]), axis=0)

    # Set a minimum variance threshold to avoid numerical instability
    min_variance = 1e-6
    feature_variances = np.maximum(feature_variances, min_variance)

    # Calculate absolute errors for each feature at each timestep
    abs_errors = np.abs(test_data - reconstructed_data)

    # Normalize errors by feature variance
    normalized_errors = abs_errors / feature_variances

    # First average across features for each timestep
    timestep_errors = np.mean(normalized_errors, axis=2)

    # Then average across timesteps for each sequence
    return np.mean(timestep_errors, axis=1)


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


def build_and_fit_feature_pipeline(df_for_fitting, all_boolean_column_names_as_int):
    """
    Builds and fits the standard feature engineering pipeline.
    Assumes boolean columns have already been identified and cast to integer in df_for_fitting.

    Args:
        df_for_fitting (DataFrame): Spark DataFrame to fit the pipeline on
                                    (e.g., normal data only for training an AE).
        all_boolean_column_names_as_int (list): List of column names that were
                                              originally boolean and are now integer type
                                              in df_for_fitting.

    Returns:
        tuple: (fitted_pipeline: PipelineModel,
                feature_cols_for_assembler: list)
    """
    print(
        f"Building and fitting pipeline on DataFrame with {df_for_fitting.count()} records."
    )

    # Infer categorical and numerical types.
    # Note: infer_column_types_from_schema might classify integer-cast booleans as numerical.
    # We will handle this by ensuring boolean columns are treated distinctly.
    categorical_cols_inferred, numerical_cols_inferred = infer_column_types_from_schema(
        df_for_fitting.schema
    )

    # Exclude known boolean columns (now int) from the inferred numerical list
    # to avoid double-counting or mis-typing them in the imputer.
    numerical_cols_for_imputer = [
        c for c in numerical_cols_inferred if c not in all_boolean_column_names_as_int
    ]

    problematic_numerical_cols = []
    for col_name in numerical_cols_for_imputer:
        if (
            df_for_fitting.where(
                col(col_name).isNotNull() & ~isnan(col(col_name))
            ).count()
            == 0
        ):
            problematic_numerical_cols.append(col_name)
            print(
                f"Warning: Numerical column '{col_name}' contains only null/NaN values "
                f"in the provided DataFrame and will be excluded from imputation and features."
            )

    valid_numerical_cols_for_imputer = [
        c for c in numerical_cols_for_imputer if c not in problematic_numerical_cols
    ]

    indexers = [
        StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
        for c in categorical_cols_inferred
    ]
    imputer = Imputer(
        inputCols=valid_numerical_cols_for_imputer,
        outputCols=valid_numerical_cols_for_imputer,
        strategy="mean",
    )

    # Construct the list of features for the assembler
    feature_cols_for_assembler_temp = (
        [c + "_idx" for c in categorical_cols_inferred]
        + valid_numerical_cols_for_imputer
        + all_boolean_column_names_as_int  # These are already int type
    )

    # Remove duplicates, preserving order
    seen = set()
    feature_cols_for_assembler = [
        x for x in feature_cols_for_assembler_temp if not (x in seen or seen.add(x))
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols_for_assembler,
        outputCol="assembled_features",
        handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol="assembled_features",
        outputCol="features",
        withMean=True,
        withStd=True,
    )

    pipeline_stages = []
    if (
        valid_numerical_cols_for_imputer
    ):  # Only add imputer if there are numerical cols to impute
        pipeline_stages.append(imputer)
    pipeline_stages.extend(indexers)
    pipeline_stages.append(assembler)
    pipeline_stages.append(scaler)

    pipeline = Pipeline(stages=pipeline_stages)
    print("Fitting pipeline...")
    fitted_pipeline = pipeline.fit(df_for_fitting)
    print("Pipeline fitting complete.")

    print(
        "\nFeature Information (from common utility function build_and_fit_feature_pipeline):"
    )
    print(
        f"  Categorical features processed (after indexing): {len(categorical_cols_inferred)}"
    )
    print(
        f"  Numerical features for imputer (valid): {len(valid_numerical_cols_for_imputer)}"
    )
    print(
        f"  Boolean features (as integer input): {len(all_boolean_column_names_as_int)}"
    )
    print(f"  Total features for VectorAssembler: {len(feature_cols_for_assembler)}")
    # print(f"  Feature columns for assembler: {feature_cols_for_assembler}") # Uncomment for debugging

    return fitted_pipeline, feature_cols_for_assembler


def plot_reconstruction_error(errors, threshold, table_name, true_labels=None):
    """
    Plots the reconstruction error, highlighting anomalies.
    If true_labels are provided, it distinguishes between True Positives and False Positives.
    """
    import os
    import matplotlib.pyplot as plt

    predicted_anomalies = errors > threshold

    plt.figure(figsize=(14, 7))

    if true_labels is not None:
        # Ensure true_labels is a boolean numpy array of the same shape as errors
        true_labels = np.array(true_labels, dtype=bool)
        if true_labels.shape != errors.shape:
            # Pad or truncate labels if there's a mismatch, which can happen
            # if the sequence creation logic is not perfectly aligned.
            # This is a fallback, the primary alignment should be done in the calling script.
            if len(true_labels) > len(errors):
                true_labels = true_labels[: len(errors)]
            else:
                padding = len(errors) - len(true_labels)
                true_labels = np.pad(
                    true_labels, (padding, 0), "constant", constant_values=(False,)
                )

        true_positives = predicted_anomalies & true_labels
        false_positives = predicted_anomalies & ~true_labels
        # All points not predicted as anomalies (True Negatives and False Negatives)
        other_points = ~predicted_anomalies

        plt.scatter(
            np.arange(len(errors))[other_points],
            errors[other_points],
            c="blue",
            label="Not Detected as Anomaly",
            s=10,
        )
        plt.scatter(
            np.arange(len(errors))[false_positives],
            errors[false_positives],
            c="red",
            label="False Positive",
            s=20,
            edgecolors="k",
        )
        plt.scatter(
            np.arange(len(errors))[true_positives],
            errors[true_positives],
            c="yellow",
            label="True Positive",
            s=20,
            edgecolors="k",
        )
    else:
        # Fallback to previous behavior if no true labels are provided
        normal_points = ~predicted_anomalies
        plt.scatter(
            np.arange(len(errors))[normal_points],
            errors[normal_points],
            c="blue",
            label="Normal",
            s=10,
        )
        plt.scatter(
            np.arange(len(errors))[predicted_anomalies],
            errors[predicted_anomalies],
            c="yellow",
            label="Anomaly",
            s=20,
            edgecolors="k",
        )

    plt.axhline(
        y=threshold,
        color="r",
        linestyle="--",
        label=f"Anomaly Threshold ({threshold:.4f})",
    )
    plt.title(f"Reconstruction Error for {table_name}")
    plt.xlabel("Data Point Index")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"reconstruction_error_{table_name}.png")
    plt.savefig(plot_path)
    print(f"Reconstruction error plot saved to: {plot_path}")
    plt.close()


def save_pipeline_and_features(
    fitted_pipeline: PipelineModel,
    feature_cols_for_assembler: list,
    table_name: str,
    output_dir: str = "./models",
):
    import os
    """
    Saves the fitted pipeline and feature columns to disk.

    Args:
        fitted_pipeline (PipelineModel): The fitted pipeline to save.
        feature_cols_for_assembler (list): List of feature columns for the assembler.
        table_name (str): The name of the table or dataset.
        output_dir (str): The directory to save the pipeline and features.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save the pipeline
    pipeline_path = os.path.join(output_dir, f"{table_name}_pipeline.pkl")
    fitted_pipeline.write().overwrite().save(pipeline_path)
    print(f"Pipeline saved to {pipeline_path}")

    # Save the feature columns
    features_path = os.path.join(output_dir, f"{table_name}_features.txt")
    with open(features_path, "w") as f:
        f.write("\n".join(feature_cols_for_assembler))
    print(f"Feature columns saved to {features_path}")
