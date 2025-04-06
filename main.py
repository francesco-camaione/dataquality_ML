from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException, ParseException
import os
import sys

# AWS Configuration
AWS_REGION = "eu-west-1"
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
S3_WAREHOUSE = os.environ.get("S3_WAREHOUSE")
LOCAL_CATALOG_PATH = os.path.abspath("iceberg_catalog")  # your local folder

spark = (
    SparkSession.builder.appName("DQ App")
    .config(
        "spark.jars.packages",
        "org.apache.iceberg:iceberg-spark-runtime-3.3_2.12:1.0.0,"
        "org.apache.hadoop:hadoop-aws:3.3.4",
    )
    .config(
        "spark.sql.extensions",
        "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
    )
    .config(
        "spark.sql.catalog.iceberg_catalog", "org.apache.iceberg.spark.SparkCatalog"
    )
    .config("spark.sql.catalog.iceberg_catalog.type", "hadoop")
    .config(
        "spark.sql.catalog.iceberg_catalog.warehouse", f"file://{LOCAL_CATALOG_PATH}"
    )
    .config("spark.sql.defaultCatalog", "iceberg_catalog")
    .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY)
    .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY)
    .config("spark.hadoop.fs.s3a.endpoint", f"s3.{AWS_REGION}.amazonaws.com")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    )
    .getOrCreate()
)

df = spark.sql("SELECT * FROM iceberg_catalog.dq.unpivoted_data LIMIT 30")

df.show()

# NEXT STEPS:
# USE A PYSPARK SESSION TO READ DATA FROM THE CSV FILE TO AVOID CONSUMING AWS
# USE THE DATA TO CONCENTRATE ON THE ML MODELS

