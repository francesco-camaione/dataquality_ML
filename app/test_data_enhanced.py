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
    input_csv = "./dataset/2024_12_25_test.csv"  # Original test dataset
    output_csv = "./dataset/2024_12_25_test_enhanced.csv"
    train_csv = "./dataset/2024_12_bb_3d.csv"  # Use training data for pipeline fitting

    # Start Spark
    connector = SparkToAWS()
    spark = connector.create_local_spark_session()

    # Load training data for pipeline fitting (use same pipeline as training)
    train_df_full = spark.read.option("header", "true").csv(train_csv, inferSchema=True)

    # Load test data
    test_df = spark.read.option("header", "true").csv(input_csv, inferSchema=True)

    # Enhanced columns selection based on Backblaze SSD SMART blog
    # Core SMART attributes that are most important for SSD health monitoring
    enhanced_columns = [
        # Basic drive info
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
        # Common SMART attributes (all drive types)
        "smart_1_normalized",  # Read Error Rate
        "smart_1_raw",
        "smart_2_normalized",  # Throughput Performance
        "smart_2_raw",
        "smart_3_normalized",  # Spin-Up Time
        "smart_3_raw",
        "smart_4_normalized",  # Start/Stop Count
        "smart_4_raw",
        "smart_5_normalized",  # Reallocated Sectors Count
        "smart_5_raw",
        "smart_7_normalized",  # Seek Error Rate
        "smart_7_raw",
        "smart_8_normalized",  # Seek Time Performance
        "smart_8_raw",
        "smart_9_normalized",  # Power-On Hours (CRITICAL - common to all drives)
        "smart_9_raw",
        "smart_10_normalized",  # Spin Retry Count
        "smart_10_raw",
        "smart_11_normalized",  # Calibration Retry Count
        "smart_11_raw",
        "smart_12_normalized",  # Power Cycle Count (CRITICAL - common to all drives)
        "smart_12_raw",
        "smart_13_normalized",  # Soft Read Error Rate
        "smart_13_raw",
        "smart_15_normalized",  # Current Pending Sector Count
        "smart_15_raw",
        "smart_16_normalized",  # Offline Uncorrectable Sector Count
        "smart_16_raw",
        "smart_17_normalized",  # Current Pending Sector Count
        "smart_17_raw",
        "smart_18_normalized",  # Offline Uncorrectable Sector Count
        "smart_18_raw",
        # SSD-SPECIFIC CRITICAL ATTRIBUTES (from Backblaze blog)
        # Lifetime and Wear Indicators
        "smart_169_normalized",  # Remaining Lifetime Percentage (Western Digital)
        "smart_169_raw",
        "smart_202_normalized",  # Percentage of Lifetime Used (Crucial)
        "smart_202_raw",
        "smart_231_normalized",  # Life Left (Seagate)
        "smart_231_raw",
        "smart_232_normalized",  # Endurance Remaining (Seagate/WD)
        "smart_232_raw",
        "smart_233_normalized",  # Media Wearout Indicator (Seagate/WD)
        "smart_233_raw",
        # SSD Wear Leveling and Power Loss
        "smart_173_normalized",  # SSD Wear Leveling (CRITICAL - common to all SSDs)
        "smart_173_raw",
        "smart_174_normalized",  # Unexpected Power Loss Count (CRITICAL - common to all SSDs)
        "smart_174_raw",
        # SSD Program/Erase Failures
        "smart_171_normalized",  # SSD Program Fail Count (WD/Crucial)
        "smart_171_raw",
        "smart_172_normalized",  # SSD Erase Count Fail (WD/Crucial)
        "smart_172_raw",
        # Temperature (CRITICAL - common to all drives)
        "smart_194_normalized",  # Temperature
        "smart_194_raw",
        # Additional SSD-specific attributes
        "smart_175_normalized",  # Program Fail Count Total
        "smart_175_raw",
        "smart_176_normalized",  # Erase Fail Count Total
        "smart_176_raw",
        "smart_177_normalized",  # Wear Leveling Count
        "smart_177_raw",
        "smart_178_normalized",  # Used Reserved Block Count Total
        "smart_178_raw",
        "smart_179_normalized",  # Used Reserved Block Count Chip
        "smart_179_raw",
        "smart_180_normalized",  # Unused Reserved Block Count Total
        "smart_180_raw",
        "smart_181_normalized",  # Program Fail Count Total
        "smart_181_raw",
        "smart_182_normalized",  # Erase Fail Count Total
        "smart_182_raw",
        "smart_183_normalized",  # SATA Downshift Error Count
        "smart_183_raw",
        "smart_184_normalized",  # End-to-End Error Detection Count
        "smart_184_raw",
        "smart_187_normalized",  # Reported Uncorrectable Errors
        "smart_187_raw",
        "smart_188_normalized",  # Command Timeout
        "smart_188_raw",
        "smart_189_normalized",  # High Fly Writes
        "smart_189_raw",
        "smart_190_normalized",  # Airflow Temperature
        "smart_190_raw",
        "smart_191_normalized",  # G-Sense Error Rate
        "smart_191_raw",
        "smart_192_normalized",  # Power-Off Retract Count
        "smart_192_raw",
        "smart_193_normalized",  # Load Cycle Count
        "smart_193_raw",
        "smart_195_normalized",  # Hardware ECC Recovered
        "smart_195_raw",
        "smart_196_normalized",  # Reallocation Event Count
        "smart_196_raw",
        "smart_197_normalized",  # Current Pending Sector Count
        "smart_197_raw",
        "smart_198_normalized",  # Uncorrectable Sector Count
        "smart_198_raw",
        "smart_199_normalized",  # UltraDMA CRC Error Count
        "smart_199_raw",
        "smart_200_normalized",  # Multi-Zone Error Rate
        "smart_200_raw",
        "smart_201_normalized",  # Soft Read Error Rate
        "smart_201_raw",
        "smart_203_normalized",  # Available Reserved Space
        "smart_203_raw",
        "smart_204_normalized",  # Soft ECC Correction
        "smart_204_raw",
        "smart_205_normalized",  # Thermal Asperity Rate
        "smart_205_raw",
        "smart_206_normalized",  # Flying Height
        "smart_206_raw",
        "smart_207_normalized",  # Spin High Current
        "smart_207_raw",
        "smart_208_normalized",  # Spin Buzz
        "smart_208_raw",
        "smart_209_normalized",  # Offline Seek Performance
        "smart_209_raw",
        "smart_220_normalized",  # Disk Shift
        "smart_220_raw",
        "smart_221_normalized",  # G-Sense Error Rate
        "smart_221_raw",
        "smart_222_normalized",  # Loaded Hours
        "smart_222_raw",
        "smart_223_normalized",  # Load/Unload Retry Count
        "smart_223_raw",
        "smart_224_normalized",  # Load Friction
        "smart_224_raw",
        "smart_225_normalized",  # Load/Unload Cycle Count
        "smart_225_raw",
        "smart_226_normalized",  # Load-In Time
        "smart_226_raw",
        "smart_227_normalized",  # Torque Amplification Count
        "smart_227_raw",
        "smart_228_normalized",  # Power-Off Retract Count
        "smart_228_raw",
        "smart_230_normalized",  # GMR Head Amplitude
        "smart_230_raw",
        "smart_240_normalized",  # Head Flying Hours
        "smart_240_raw",
        "smart_241_normalized",  # Total LBAs Written
        "smart_241_raw",
        "smart_242_normalized",  # Total LBAs Read
        "smart_242_raw",
        "smart_250_normalized",  # Read Error Retry Rate
        "smart_250_raw",
        "smart_251_normalized",  # Free Fall Protection
        "smart_251_raw",
        "smart_252_normalized",  # Free Fall Protection
        "smart_252_raw",
        "smart_254_normalized",  # Free Fall Protection
        "smart_254_raw",
        "smart_255_normalized",  # Free Fall Protection
        "smart_255_raw",
    ]

    # This ensures feature consistency between training and test
    train_columns = train_df_full.columns

    # Filter enhanced_columns to only include those available in the training data
    available_columns = [col for col in enhanced_columns if col in train_columns]

    # Add missing columns to the test dataframe and fill with null
    # This ensures the test data has the same schema as the training data for the pipeline
    for column in available_columns:
        if column not in test_df.columns:
            test_df = test_df.withColumn(column, lit(None).cast(DoubleType()))

    print(f"Training dataset has {len(train_columns)} columns")
    print(f"Test dataset has {len(test_df.columns)} columns after alignment")
    print(f"Enhanced selection includes {len(available_columns)} columns")
    print(
        f"Missing enhanced columns from training data: {set(enhanced_columns) - set(train_columns)}"
    )

    # Select only the aligned columns for both dataframes
    train_df_full = train_df_full.select(*available_columns)
    test_df = test_df.select(*available_columns)

    # Boolean columns handling
    bool_cols = boolean_columns(train_df_full.schema)
    if bool_cols:
        for column in bool_cols:
            train_df_full = train_df_full.withColumn(
                column, col(column).cast(types.IntegerType())
            )
            test_df = test_df.withColumn(column, col(column).cast(types.IntegerType()))

    # Fit pipeline on normal data from training set (same as training)
    train_df_normal = train_df_full.where(col("failure") == 0)
    fitted_pipeline, feature_cols = build_and_fit_feature_pipeline(
        train_df_normal, bool_cols
    )

    # Preprocess test data using the same pipeline
    processed_test_df = fitted_pipeline.transform(test_df)
    processed_pd = processed_test_df.select(["features", "failure"]).toPandas()

    # Convert features vector to columns
    features_array = processed_pd["features"].apply(lambda x: x.toArray()).tolist()
    features_df = pd.DataFrame(features_array, columns=feature_cols)
    features_df["failure"] = processed_pd["failure"].values

    df_normal = features_df[features_df["failure"] == 0]
    df_failure = features_df[features_df["failure"] == 1]

    print(f"Original test data - Normal: {len(df_normal)}, Failures: {len(df_failure)}")

    # Target: 50k normal, 10k failure
    target_normal = 50000
    target_failure = 10000

    # Sample normal data to 50k
    if len(df_normal) >= target_normal:
        df_normal_sampled = df_normal.sample(n=target_normal, random_state=42)
    else:
        # If we don't have enough normal data, duplicate with replacement
        print(
            f"Warning: Only {len(df_normal)} normal samples available, duplicating to reach {target_normal}"
        )
        df_normal_sampled = df_normal.sample(
            n=target_normal, random_state=42, replace=True
        )

    # Sample failure data to 10k
    if len(df_failure) >= target_failure:
        df_failure_sampled = df_failure.sample(n=target_failure, random_state=42)
    else:
        # If we don't have enough failure data, use SMOTE to generate more
        print(
            f"Warning: Only {len(df_failure)} failure samples available, using SMOTE to reach {target_failure}"
        )

        # Combine normal and failure data for SMOTE
        df_combined = pd.concat([df_normal_sampled, df_failure], ignore_index=True)
        X = df_combined.drop(columns=["failure"])
        y = df_combined["failure"]

        # Apply SMOTE to generate the needed number of failure samples
        smote = SMOTE(
            random_state=42,
            k_neighbors=min(5, len(df_failure) - 1) if len(df_failure) > 1 else 1,
            sampling_strategy={1: target_failure},
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Extract the failure samples from resampled data
        failure_mask = y_resampled == 1
        df_failure_sampled = pd.DataFrame(
            X_resampled[failure_mask], columns=feature_cols
        )
        df_failure_sampled["failure"] = 1

        # Keep the original normal samples
        normal_mask = y_resampled == 0
        df_normal_sampled = pd.DataFrame(X_resampled[normal_mask], columns=feature_cols)
        df_normal_sampled["failure"] = 0

    # Combine to create balanced test dataset
    df_final = pd.concat([df_normal_sampled, df_failure_sampled], ignore_index=True)

    # Shuffle the final dataset
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    print(
        f"Final enhanced test dataset - Normal: {len(df_final[df_final['failure'] == 0])}, Failures: {len(df_final[df_final['failure'] == 1])}"
    )
    print(f"Total test records: {len(df_final)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")

    df_final.to_csv(output_csv, index=False)
    print(f"Enhanced test dataset saved to {output_csv} with shape {df_final.shape}")
    print(
        f"Test dataset includes critical SSD SMART attributes for better failure prediction"
    )
    print(
        f"Test dataset ratio: {len(df_final[df_final['failure'] == 0])} normal : {len(df_final[df_final['failure'] == 1])} failure"
    )

    connector.close_spark_session()


if __name__ == "__main__":
    main()
