import tensorflow as tf
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when
from pyspark.sql.types import StructType  # Needed for type hints

# Assuming align_dataframe_schema is in spark_utils
from .spark_utils import align_dataframe_schema


def preprocess_data(df: DataFrame, reference_schema: StructType = None):
    """Removes fully null columns and aligns to a reference schema if provided."""
    # Get columns for null check (exclude potential target and ID columns)
    # Make this more robust by checking types or using a predefined list if needed
    cols_to_check = [c for c in df.columns if c not in ["failure", "record_id"]]

    if not cols_to_check:
        print("Warning: No columns found to check for nulls in preprocess_data.")
        if reference_schema:
            print("Aligning schema based on reference.")
            return align_dataframe_schema(df, reference_schema)
        else:
            return df

    # Check for fully null columns
    total_rows = df.count()
    if total_rows == 0:
        print("Warning: DataFrame is empty in preprocess_data.")
        if reference_schema:
            print("Aligning schema of empty DataFrame based on reference.")
            return align_dataframe_schema(df, reference_schema)
        else:
            return df  # Return empty DF

    # Efficiently check for all nulls
    null_counts_expr = [count(when(col(c).isNull(), c)).alias(c) for c in cols_to_check]
    null_counts = df.select(null_counts_expr).first()

    all_null_cols = [c for c in cols_to_check if null_counts[c] == total_rows]

    df_cleaned = df
    if all_null_cols:
        print(
            f"Removing {len(all_null_cols)} columns that are all null: {all_null_cols}"
        )
        df_cleaned = df.drop(*all_null_cols)

    # Align to reference schema if provided
    if reference_schema:
        print(
            f"Aligning DataFrame to reference schema ({len(reference_schema.fields)} fields)."
        )
        df_aligned = align_dataframe_schema(df_cleaned, reference_schema)
        return df_aligned
    else:
        print("No reference schema provided, returning cleaned DataFrame.")
        return df_cleaned


def process_batch_to_tensor(df, fitted_pipeline):
    """Transforms Spark DataFrame features to a TensorFlow tensor using an iterator to handle large data."""
    print("Transforming DataFrame and preparing iterator...")
    # Transform and select features and record_id
    processed_df = fitted_pipeline.transform(df).select("features", "record_id")

    # Use toLocalIterator() instead of collect() for memory efficiency
    # Warning: This still brings data partition by partition to the driver.
    # For extremely large datasets distributed processing (e.g., Petastorm, tf.data.Dataset.from_generator with Spark UDFs)
    # might be necessary.
    try:
        data_iterator = processed_df.toLocalIterator()
    except Exception as e:
        print(f"Error creating local iterator: {e}")
        return None, []  # Indicate failure

    features_list = []
    record_ids = []
    num_features = -1  # Initialize feature count
    rows_processed = 0

    print("Iterating through partitions and collecting data...")
    try:
        for row in data_iterator:
            # Assuming 'features' column contains VectorUDT
            feature_array = row.features.toArray()
            features_list.append(feature_array)
            record_ids.append(row.record_id)
            rows_processed += 1

            # Infer num_features from the first row
            if num_features == -1:
                num_features = len(feature_array)
                print(f"Inferred number of features: {num_features}")

            # Optional: Add logging for progress
            # if rows_processed % 10000 == 0:
            #     print(f"Processed {rows_processed} rows...")

    except Exception as e:
        print(f"Error during iteration: {e}")
        # Decide how to handle partial results, here we return what we got
        if not features_list:
            print("No features collected before error.")
            return None, []

    print(f"Finished iterating. Total rows processed: {rows_processed}")

    if not features_list:
        print("Warning: No data collected after iterating through partitions.")
        # Try to infer num_features from pipeline schema if possible, even if no rows
        if num_features == -1:  # If we didn't even process one row
            try:
                feature_vector_type = (
                    fitted_pipeline.stages[-1]
                    .transformSchema(processed_df.schema)["features"]
                    .dataType
                )
                num_features = feature_vector_type.size
                print(f"Inferred {num_features} features for empty tensor from schema.")
            except Exception:
                print(
                    "Warning: Could not infer feature size for empty tensor. Returning None."
                )
                return None, []

        # Return empty tensor with correct feature dimension if known
        if num_features != -1:
            return tf.zeros((0, num_features), dtype=tf.float32), []
        else:
            return None, []  # Cannot determine shape

    # Convert the collected list to TensorFlow tensor
    try:
        tensor = tf.convert_to_tensor(features_list, dtype=tf.float32)
        print(f"Converted collected data to tensor with shape: {tensor.shape}")
        return tensor, record_ids
    except Exception as e:
        print(f"Error converting features list to tensor: {e}")
        # Consider returning partial data or None
        return None, record_ids  # Indicate conversion failure
