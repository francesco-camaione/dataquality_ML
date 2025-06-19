import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import pandas as pd
from imblearn.over_sampling import SMOTE
from lib.connector import SparkToAWS
from lib.utils import boolean_columns, build_and_fit_feature_pipeline
from pyspark.sql.functions import col, lit
from pyspark.sql import types


def main():
    input_csv = "./dataset/2024_12_bb_3d.csv"  # Training dataset
    output_csv = "./dataset/2024_12_bb_3d_augmented.csv"
    train_csv = input_csv  # For pipeline fitting, use the same file

    # Start Spark
    connector = SparkToAWS()
    spark = connector.create_local_spark_session()

    # Load full training data for schema and pipeline fitting
    train_df_full = spark.read.option("header", "true").csv(train_csv, inferSchema=True)
    test_df = train_df_full  # For training, we process the same file

    # Align columns (add missing, drop extra)
    train_cols = set(train_df_full.columns)
    test_cols = set(test_df.columns)
    missing_in_test = train_cols - test_cols
    extra_in_test = test_cols - train_cols
    if missing_in_test:
        for col_name in missing_in_test:
            train_col_type = train_df_full.schema[col_name].dataType
            test_df = test_df.withColumn(col_name, lit(None).cast(train_col_type))
    if extra_in_test:
        test_df = test_df.drop(*extra_in_test)

    # Boolean columns handling
    bool_cols = boolean_columns(train_df_full.schema)
    if bool_cols:
        for column in bool_cols:
            train_df_full = train_df_full.withColumn(
                column, col(column).cast(types.IntegerType())
            )
            test_df = test_df.withColumn(column, col(column).cast(types.IntegerType()))

    # Fit pipeline on normal data from training set
    train_df_normal = train_df_full.where(col("failure") == 0)
    fitted_pipeline, feature_cols = build_and_fit_feature_pipeline(
        train_df_normal, bool_cols
    )

    # Preprocess training data
    processed_test_df = fitted_pipeline.transform(test_df)
    processed_pd = processed_test_df.select(["features", "failure"]).toPandas()

    # Convert features vector to columns
    features_array = processed_pd["features"].apply(lambda x: x.toArray()).tolist()
    features_df = pd.DataFrame(features_array, columns=feature_cols)
    features_df["failure"] = processed_pd["failure"].values

    df_normal = features_df[features_df["failure"] == 0]
    df_failure = features_df[features_df["failure"] == 1]
    df_normal_reduced = df_normal.sample(n=250000, random_state=42)

    # Combine for SMOTE
    df_combined = pd.concat([df_normal_reduced, df_failure], ignore_index=True)
    X = df_combined.drop(columns=["failure"])
    y = df_combined["failure"]

    # Apply SMOTE with robust k_neighbors
    n_minority = df_failure.shape[0]
    k_neighbors = min(5, n_minority - 1) if n_minority > 1 else 1
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_final = pd.DataFrame(X_resampled, columns=feature_cols)
    df_final["failure"] = y_resampled

    df_final.to_csv(output_csv, index=False)
    print(
        f"Augmented training dataset saved to {output_csv} with shape {df_final.shape}"
    )

    connector.close_spark_session()


if __name__ == "__main__":
    main()
