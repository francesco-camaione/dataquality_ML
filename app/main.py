from pyspark.sql import SparkSession
import os

# AWS Configuration
AWS_REGION = "eu-west-1"
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
S3_WAREHOUSE = os.environ.get("S3_WAREHOUSE")
GLUE_CATALOG_ID = os.environ.get("GLUE_CATALOG_ID")

spark = (
    SparkSession.builder.appName("DQ App")
    .config(
        "spark.jars.packages",
        "org.apache.iceberg:iceberg-spark-runtime-3.3_2.12:1.8.1,"
        "org.apache.iceberg:iceberg-aws-bundle:1.8.1,"
        "org.apache.hadoop:hadoop-aws:3.3.4,"
        "software.amazon.awssdk:url-connection-client:2.20.160",
    )
    .config("spark.jars.repositories", "https://repo1.maven.org/maven2/,https://repos.spark-packages.org/,https://repo.maven.apache.org/maven2/",)
    .config("spark.sql.catalog.s3tablescatalog", "org.apache.iceberg.spark.SparkCatalog",)
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.s3tablescatalog.glueRegion", AWS_REGION)
    .config("spark.sql.catalog.s3tablescatalog.glue.catalog-id", GLUE_CATALOG_ID)
    .config("spark.sql.catalog.s3tablescatalog.type", "glue")
    .config("spark.sql.catalog.s3tablescatalog.warehouse", S3_WAREHOUSE)
    .config("spark.sql.catalog.s3tablescatalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY)
    .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY)
    .config("spark.hadoop.fs.s3a.endpoint", f"s3.{AWS_REGION}.amazonaws.com")
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    .getOrCreate()
)

print("Spark is running...")

# spark.sql("""
#     CREATE OR REPLACE TABLE s3tablescatalog.dq_db.unpivoted_datacenter_data
#     USING iceberg
#     LOCATION 's3a://dq-monitoring-bucket/iceberg/'
#     AS SELECT * 
#     FROM parquet.`s3a://dq-monitoring-bucket/raw_data/unpivoted_data.parquet`;
# """).show()

spark.sql("SELECT COUNT(*) AS tot FROM s3tablescatalog.dq_db.unpivoted_datacenter_data;").show()

print("Spark is done!!")