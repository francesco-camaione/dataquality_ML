import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from lib.connector import SparkToAWS


connector = SparkToAWS()
spark_session = connector.create_spark_session()

print("Spark is running...")

# df = spark_session.read.option("header", "true").option("inferSchema", "true").csv("s3a://dq-monitoring-bucket/raw_data/2024_12_15_test.csv")
# df.writeTo("s3tablescatalog.dq_db.2024_12_15_test").using("iceberg").createOrReplace()

spark_session.sql(
    """
       SELECT *
       FROM s3tablescatalog.dq_db.2024_12_bb_3d
       LIMIT 10;
   """
).show()

# purge when dropping the table to remove the orphan metadata files
# spark_session.sql(f"""DROP TABLE s3tablescatalog.dq_db.{table_name} PURGE;""").show()

connector.close_spark_session()
print("Spark is done!!")
