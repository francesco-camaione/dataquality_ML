from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    StructType,
)  # Needed for type hint in align_dataframe_schema


def create_spark_session(config):
    builder = SparkSession.builder.appName("LSTM_AE_Backblaze").master("local[*]")
    for key, value in config.items():
        builder = builder.config(key, value)
    print("Spark session created/retrieved.")
    return builder.getOrCreate()


def align_dataframe_schema(df_to_align, reference_schema: StructType):
    """Aligns df_to_align schema to reference_schema, adding nulls for missing columns."""
    cols_to_keep = {f.name for f in reference_schema.fields}
    existing_cols = {c for c in df_to_align.columns}

    select_expr = []
    added_cols = []
    aligned_cols = []

    for field in reference_schema.fields:
        col_name = field.name
        col_type = field.dataType
        if col_name in existing_cols:
            # Cast existing column to the reference type
            select_expr.append(col(col_name).cast(col_type))
            aligned_cols.append(col_name)
        else:
            # Add missing column as null literal with the reference type
            select_expr.append(lit(None).cast(col_type).alias(col_name))
            added_cols.append(col_name)

    if added_cols:
        print(f"Added missing columns with nulls: {added_cols}")
    # print(f"Aligning existing columns: {aligned_cols}") # Might be too verbose

    return df_to_align.select(select_expr)
