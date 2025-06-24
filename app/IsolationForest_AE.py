import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import gc
import pickle
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.sql.functions import col, isnan
from pyspark.sql.types import NumericType
from lib.connector import SparkToAWS
from lib.utils import plot_roc_curve, plot_reconstruction_error
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score


def preprocess_data(spark, train_table_name_arg, test_table_name_arg):
    """
    Loads and preprocesses both training and test data.
    Fits preprocessing pipeline on training data and applies it to both datasets.
    """
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

    train_df = (
        spark.read.option("header", "true")
        .csv(f"./dataset/{train_table_name_arg}.csv", inferSchema=True)
        .select(*columns_to_select_initial)
    )

    test_df = (
        spark.read.option("header", "true")
        .csv(f"./dataset/{test_table_name_arg}.csv", inferSchema=True)
        .select(*columns_to_select_initial)
    )

    all_numeric_cols = [
        f.name
        for f in train_df.schema.fields
        if isinstance(f.dataType, NumericType) and f.name.lower() != "failure"
    ]

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
    pipeline_save_path = f"./pipelines/IF_pipeline/IF_pipeline_{train_table_name_arg}"
    pipeline_model.write().overwrite().save(pipeline_save_path)
    print(f"Full preprocessing pipeline model saved to: {pipeline_save_path}")

    # Transform both training and test data
    print("Transforming training data...")
    processed_train_df = pipeline_model.transform(train_df)
    print("Transforming test data...")
    processed_test_df = pipeline_model.transform(test_df)

    # Convert to pandas and prepare data
    columns_to_select = ["features"]
    if "failure" in processed_train_df.columns:
        columns_to_select.append("failure")

    # Process training data
    train_pdf = processed_train_df.select(columns_to_select).toPandas()
    train_feature_list = (
        train_pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
    )
    X_train = np.array(train_feature_list, dtype=np.float32)

    # Process test data
    test_pdf = processed_test_df.select(columns_to_select).toPandas()
    test_feature_list = (
        test_pdf["features"].apply(lambda x: np.array(x.toArray())).tolist()
    )
    X_test = np.array(test_feature_list, dtype=np.float32)

    # Prepare labels if failure column exists
    y_train = None
    y_test = None
    if "failure" in train_pdf.columns:
        y_train = train_pdf["failure"].values
        y_test = test_pdf["failure"].values

    return (
        train_df.toPandas(),
        test_df.toPandas(),
        X_train,
        X_test,
        y_train,
        y_test,
    )


def load_or_create_model(model_path_arg, X_train_arg, y_train_arg=None):
    """
    Loads an existing model or creates and trains a new one with parameter tuning.
    """
    model_dir = os.path.dirname(model_path_arg)
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(model_path_arg):
        print(f"Loading existing model from {model_path_arg}")
        with open(model_path_arg, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    else:
        print("Training new Isolation Forest model with parameter tuning...")
        print(f"Training data shape: {X_train_arg.shape}")
        print(f"Number of features: {X_train_arg.shape[1]}")
        print(f"Number of samples: {X_train_arg.shape[0]}")

        # Define parameter grid for tuning
        param_grid = {
            "n_estimators": [4000, 6000, 8000],
            "max_samples": [8, 16, 32],
            "contamination": [0.0001, 0.00025, 0.0005],
            "max_features": [0.1, 0.2, 0.3],
            "bootstrap": [True, False],
        }

        # Initialize base model
        base_model = IsolationForest(random_state=42, n_jobs=-1)

        # Create custom scorer for anomaly detection
        def anomaly_scorer(y_true, y_pred):
            # Convert anomaly scores to binary predictions using a threshold
            threshold = np.percentile(y_pred, 72)  # Using 72nd percentile as threshold
            y_pred_binary = y_pred > threshold
            return roc_auc_score(y_true, y_pred_binary)

        custom_scorer = make_scorer(anomaly_scorer)

        # Perform grid search with cross-validation
        print("Starting parameter tuning with cross-validation...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=custom_scorer,
            cv=5,
            n_jobs=-1,
            verbose=2,
        )

        # Fit the grid search
        import time

        start_time = time.time()
        grid_search.fit(X_train_arg)
        training_time = time.time() - start_time
        print(f"Parameter tuning completed in {training_time:.2f} seconds")

        # Get best parameters and model
        best_params = grid_search.best_params_
        print("\nBest parameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Initialize final model with best parameters
        model = IsolationForest(**best_params, random_state=42, n_jobs=-1)

        print("Training final model with best parameters...")
        model.fit(X_train_arg)

        # Verify model training
        print("Verifying model training...")
        train_scores = model.score_samples(X_train_arg)
        print(
            f"Training scores - Min: {np.min(train_scores):.4f}, Max: {np.max(train_scores):.4f}, Mean: {np.mean(train_scores):.4f}"
        )

        # Save the model
        print("Saving model...")
        with open(model_path_arg, "wb") as f:
            pickle.dump(model, f)
        print(f"New model saved to {model_path_arg}")

    return model


def predict_and_evaluate(model, X_test_arg):
    """
    Predicts and evaluates the model on test data.
    """
    print(f"Predicting on test data with shape: {X_test_arg.shape}")
    # Get anomaly scores (negative scores indicate anomalies)
    anomaly_scores = -model.score_samples(X_test_arg)
    print(f"Generated {len(anomaly_scores)} anomaly scores")
    print(
        f"Score range - Min: {np.min(anomaly_scores):.4f}, Max: {np.max(anomaly_scores):.4f}"
    )

    return anomaly_scores


def main():
    train_table_name = "2024_12_bb_3d"
    test_table_name = "2024_12_25_test"

    connector = SparkToAWS()
    spark_main = connector.create_local_spark_session()

    (
        original_train_pdf,
        original_test_pdf,
        X_train_main,
        X_test_main,
        y_train_main,
        y_test_main,
    ) = preprocess_data(spark_main, train_table_name, test_table_name)

    model_path_main = os.path.join("./ml_models", f"IF_{train_table_name}.pkl")

    # Check if model was loaded or newly trained
    model_was_loaded = os.path.exists(model_path_main)
    isolation_forest = load_or_create_model(model_path_main, X_train_main, y_train_main)

    del X_train_main
    gc.collect()
    print("Training data cleared.")

    anomaly_scores = predict_and_evaluate(isolation_forest, X_test_main)

    # Calculate threshold at 72nd percentile (balanced sensitivity)
    threshold = np.percentile(anomaly_scores, 90)
    print(f"\nAnomaly Threshold (72nd percentile of scores): {threshold:.4f}")

    # Save threshold
    thresholds_dir = "./dataset/thresholds"
    os.makedirs(thresholds_dir, exist_ok=True)
    threshold_file = os.path.join(
        thresholds_dir, f"IF_threshold_{train_table_name}.txt"
    )
    with open(threshold_file, "w") as f:
        f.write(f"{threshold}")
    print(f"Threshold saved to: {threshold_file}")

    # --- Anomaly Score Statistics for Test Dataset ---
    print("\n--- Anomaly Score Statistics for Test Dataset ---")
    print(f"Min Score: {np.min(anomaly_scores):.4f}")
    print(f"Max Score: {np.max(anomaly_scores):.4f}")
    print(f"Mean Score: {np.mean(anomaly_scores):.4f}")
    print(f"Median Score: {np.median(anomaly_scores):.4f}")
    print(f"Std Dev of Score: {np.std(anomaly_scores):.4f}")
    print("--------------------------------------------------\n")

    # Plot anomaly scores
    plot_reconstruction_error(
        anomaly_scores, threshold, test_table_name, true_labels=y_test_main
    )

    # Generate anomaly predictions
    anomaly_predictions = anomaly_scores > threshold

    # Create results DataFrame
    results_df = original_test_pdf.copy()
    results_df["anomaly_score"] = anomaly_scores
    results_df["is_anomaly"] = anomaly_predictions
    if y_test_main is not None:
        results_df["actual_failure"] = y_test_main

    # Save all anomalies
    output_dir = "./dataset/anomalies"
    os.makedirs(output_dir, exist_ok=True)
    anomalies_path = os.path.join(output_dir, f"IF_anomalies_{test_table_name}.csv")
    anomalies_df = results_df[results_df["is_anomaly"] == True].copy()
    anomalies_df.to_csv(anomalies_path, index=False)
    print(f"Found {len(anomalies_df)} anomalies. Saved to: {anomalies_path}")

    if y_test_main is not None:
        # --- Performance Statistics ---
        eval_df = results_df.copy()
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
            (true_positives / total_actual_failures) * 100
            if total_actual_failures > 0
            else 0
        )

        print("\n--- Model Performance Statistics ---")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"Total actual failures in evaluated data: {total_actual_failures}")
        print(f"Detection Rate: {detection_rate:.2f}%")
        print("------------------------------------\n")

        # Plot ROC curve
        plot_roc_curve(y_test_main, anomaly_scores, test_table_name)

    connector.close_spark_session()
    print("Script finished.")


if __name__ == "__main__":
    main()
