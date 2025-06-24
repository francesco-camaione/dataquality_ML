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
from pyspark.sql.types import DoubleType


def main():
    input_csv = "./dataset/2024_12_bb_3d.csv"
    output_csv = "./dataset/2024_12_bb_3d_enhanced.csv"
    train_csv = input_csv

    connector = SparkToAWS()
    spark = connector.create_local_spark_session()

    train_df_full = spark.read.option("header", "true").csv(train_csv, inferSchema=True)
    test_df = train_df_full

    enhanced_columns = [
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
        "smart_169_normalized",
        "smart_169_raw",
        "smart_202_normalized",
        "smart_202_raw",
        "smart_231_normalized",
        "smart_231_raw",
        "smart_232_normalized",
        "smart_232_raw",
        "smart_233_normalized",
        "smart_233_raw",
        "smart_173_normalized",
        "smart_173_raw",
        "smart_174_normalized",
        "smart_174_raw",
        "smart_171_normalized",
        "smart_171_raw",
        "smart_172_normalized",
        "smart_172_raw",
        "smart_194_normalized",  # Temperature
        "smart_194_raw",
        "smart_175_normalized",
        "smart_175_raw",
        "smart_176_normalized",
        "smart_176_raw",
        "smart_177_normalized",
        "smart_177_raw",
        "smart_178_normalized",
        "smart_178_raw",
        "smart_179_normalized",
        "smart_179_raw",
        "smart_180_normalized",
        "smart_180_raw",
        "smart_181_normalized",
        "smart_181_raw",
        "smart_182_normalized",
        "smart_182_raw",
        "smart_183_normalized",
        "smart_183_raw",
        "smart_184_normalized",
        "smart_184_raw",
        "smart_187_normalized",
        "smart_187_raw",
        "smart_188_normalized",
        "smart_188_raw",
        "smart_189_normalized",
        "smart_189_raw",
        "smart_190_normalized",
        "smart_190_raw",
        "smart_191_normalized",
        "smart_191_raw",
        "smart_192_normalized",
        "smart_192_raw",
        "smart_193_normalized",
        "smart_193_raw",
        "smart_195_normalized",
        "smart_195_raw",
        "smart_196_normalized",
        "smart_196_raw",
        "smart_197_normalized",
        "smart_197_raw",
        "smart_198_normalized",
        "smart_198_raw",
        "smart_199_normalized",
        "smart_199_raw",
        "smart_200_normalized",
        "smart_200_raw",
        "smart_201_normalized",
        "smart_201_raw",
        "smart_203_normalized",
        "smart_203_raw",
        "smart_204_normalized",
        "smart_204_raw",
        "smart_205_normalized",
        "smart_205_raw",
        "smart_206_normalized",
        "smart_206_raw",
        "smart_207_normalized",
        "smart_207_raw",
        "smart_208_normalized",
        "smart_208_raw",
        "smart_209_normalized",
        "smart_209_raw",
        "smart_220_normalized",
        "smart_220_raw",
        "smart_221_normalized",
        "smart_221_raw",
        "smart_222_normalized",
        "smart_222_raw",
        "smart_223_normalized",
        "smart_223_raw",
        "smart_224_normalized",
        "smart_224_raw",
        "smart_225_normalized",
        "smart_225_raw",
        "smart_226_normalized",
        "smart_226_raw",
        "smart_227_normalized",
        "smart_227_raw",
        "smart_228_normalized",
        "smart_228_raw",
        "smart_230_normalized",
        "smart_230_raw",
        "smart_240_normalized",
        "smart_240_raw",
        "smart_241_normalized",
        "smart_241_raw",
        "smart_242_normalized",
        "smart_242_raw",
        "smart_250_normalized",
        "smart_250_raw",
        "smart_251_normalized",
        "smart_251_raw",
        "smart_252_normalized",
        "smart_252_raw",
        "smart_254_normalized",
        "smart_254_raw",
        "smart_255_normalized",
        "smart_255_raw",
    ]

    for column in enhanced_columns:
        if column not in train_df_full.columns:
            train_df_full = train_df_full.withColumn(
                column, lit(None).cast(DoubleType())
            )
        if column not in test_df.columns:
            test_df = test_df.withColumn(column, lit(None).cast(DoubleType()))

    train_df_full = train_df_full.select(*enhanced_columns)
    test_df = test_df.select(*enhanced_columns)

    bool_cols = boolean_columns(train_df_full.schema)
    if bool_cols:
        for column in bool_cols:
            train_df_full = train_df_full.withColumn(
                column, col(column).cast(types.IntegerType())
            )
            test_df = test_df.withColumn(column, col(column).cast(types.IntegerType()))

    train_df_normal = train_df_full.where(col("failure") == 0)
    fitted_pipeline, feature_cols = build_and_fit_feature_pipeline(
        train_df_normal, bool_cols
    )

    processed_test_df = fitted_pipeline.transform(test_df)
    processed_pd = processed_test_df.select(["features", "failure"]).toPandas()

    features_array = processed_pd["features"].apply(lambda x: x.toArray()).tolist()
    features_df = pd.DataFrame(features_array, columns=feature_cols)
    features_df["failure"] = processed_pd["failure"].values

    df_normal = features_df[features_df["failure"] == 0]
    df_failure = features_df[features_df["failure"] == 1]

    print(f"Original data - Normal: {len(df_normal)}, Failures: {len(df_failure)}")

    target_count = 100000
    failure_count = 5000

    if len(df_normal) >= target_count:
        df_normal_sampled = df_normal.sample(n=target_count, random_state=42)
    else:
        print(
            f"Warning: Only {len(df_normal)} normal samples available, duplicating to reach {target_count}"
        )
        df_normal_sampled = df_normal.sample(
            n=target_count, random_state=42, replace=True
        )

    if len(df_failure) >= failure_count:
        df_failure_sampled = df_failure.sample(n=failure_count, random_state=42)
    else:
        print(
            f"Warning: Only {len(df_failure)} failure samples available, using SMOTE to reach {failure_count}"
        )

        df_combined = pd.concat([df_normal_sampled, df_failure], ignore_index=True)
        X = df_combined.drop(columns=["failure"])
        y = df_combined["failure"]

        smote = SMOTE(
            random_state=42,
            k_neighbors=min(5, len(df_failure) - 1) if len(df_failure) > 1 else 1,
            sampling_strategy={1: failure_count},
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)

        failure_mask = y_resampled == 1
        df_failure_sampled = pd.DataFrame(
            X_resampled[failure_mask], columns=feature_cols
        )
        df_failure_sampled["failure"] = 1

        normal_mask = y_resampled == 0
        df_normal_sampled = pd.DataFrame(X_resampled[normal_mask], columns=feature_cols)
        df_normal_sampled["failure"] = 0

    df_final = pd.concat([df_normal_sampled, df_failure_sampled], ignore_index=True)

    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    print(
        f"Final balanced dataset - Normal: {len(df_final[df_final['failure'] == 0])}, Failures: {len(df_final[df_final['failure'] == 1])}"
    )
    print(f"Total records: {len(df_final)}")
    print(f"Number of features: {len(feature_cols)}")

    df_final.to_csv(output_csv, index=False)
    print(
        f"Enhanced balanced training dataset saved to {output_csv} with shape {df_final.shape}"
    )

    connector.close_spark_session()


if __name__ == "__main__":
    main()
