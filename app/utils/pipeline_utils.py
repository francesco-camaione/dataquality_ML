from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.sql.types import StructType  # Needed for type hint
from lib.utils import infer_column_types_from_schema  # This is an existing project lib


def create_feature_pipeline(schema: StructType):
    """Creates a feature engineering pipeline for Spark ML."""
    # Remove failure and record_id before inferring types for features
    schema_for_inference = StructType(
        [f for f in schema.fields if f.name not in ["failure", "record_id"]]
    )
    categorical_cols, numerical_cols = infer_column_types_from_schema(
        schema_for_inference
    )

    print(
        f"Creating feature pipeline with {len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns."
    )
    # print(f"Categorical: {categorical_cols}") # For debugging
    # print(f"Numerical: {numerical_cols}")   # For debugging

    indexers = [
        StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
        for c in categorical_cols
    ]
    feature_cols = [c + "_idx" for c in categorical_cols] + numerical_cols

    if not feature_cols:
        print(
            "Warning: No feature columns identified for the pipeline. Assembler might fail."
        )
        # Depending on desired behavior, either raise error or return a dummy/empty pipeline
        # For now, let it proceed, Spark will error out if assembler has no inputCols

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="assembled_features", handleInvalid="keep"
    )
    scaler = StandardScaler(
        inputCol="assembled_features", outputCol="features", withMean=True, withStd=True
    )
    pipeline = Pipeline(stages=indexers + [assembler, scaler])
    print("Feature pipeline created.")
    return pipeline
