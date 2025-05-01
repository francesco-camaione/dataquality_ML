from pyspark.sql import DataFrame
import os
import requests
import json


class ColumnClassifier:
    def __init__(self, df: DataFrame, hf_token: str, sample_size: int = 100, llm_enabled: bool = True):
        self.df = df
        self.sample_size = sample_size
        self.llm_enabled = llm_enabled

        self.hf_url = "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-1"
        self.hf_token = hf_token

    def _get_sample_values(self, col_name):
        if col_name:
            return [row[col_name] for row in self.df.select(col_name).limit(self.sample_size).collect()]
        return []

    def _is_unix_timestamp(self, values):
        """
        Check if values in the numerical column resemble Unix timestamps.
        A Unix timestamp typically has 10 digits and represents seconds since 1970-01-01.
        """
        for value in values:
            if isinstance(value, int) and len(str(value)) == 10:
                return True
        return False

    def _classify_numerical_column(self, name, dtype, values):
        """
        Classify the numerical column by checking if it should be treated as a Unix timestamp.
        """
        if self._is_unix_timestamp(values):
            return "datetime"
        else:
            return "numerical"

    def classify_numerical_columns(self) -> dict:
        to_cast = []

        for field in self.df.schema.fields:
            name = field.name
            dtype = field.dataType.simpleString()

            if dtype in ['int', 'long', 'float', 'double']:  # Only handle numerical columns
                values = self._get_sample_values(name)

                # If there are values and LLM is enabled, classify with the LLM, else check for Unix timestamp
                if self.llm_enabled and values:
                    classified_type = self._classify_with_llm(name, dtype, values)
                else:
                    classified_type = self._classify_numerical_column(name, dtype, values)

                # Classify based on the determined type
                if classified_type == "datetime":
                    to_cast.append((name, "datetime"))
                # Else leave as numerical column (no action needed)

        return {"to_cast": to_cast}

    def _classify_with_llm(self, name, dtype, values):
        input_text = self._build_model_context(name, dtype, values)

        payload = {
            "inputs": input_text,
            "parameters": {
                "candidate_labels": ["date", "datetime"]
            }
        }

        headers = {
            'Authorization': f'Bearer {self.hf_token}',
            'Content-Type': 'application/json'
        }

        response = requests.post(self.hf_url, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("labels", ["unknown"])[0]
        else:
            print(f"[ERROR] Column: '{name}' - Status: {response.status_code} - Response: {response.text}")
            return "unknown"

    def _build_model_context(self, name, dtype, values):
        return f"""
            Task: Determine the semantic type of a data column.

            Instructions:
            - If the datatype is 'string', classify as 'categorical'.
            - If the datatype is 'numerical':
                - Check if values resemble Unix timestamps (i.e., 10-digit integers that represent seconds since Jan 1, 1970). If so, classify as 'datetime'.
                - Otherwise, classify as 'numerical'.

            Column:
            {{
                "name": "{name}",
                "type": "{dtype}",
                "sample_values": [{", ".join(f'"{str(v)}"' for v in values[:3])}]
            }}
        """

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from dotenv import load_dotenv
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    from lib import utils
    
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")

    spark = SparkSession.builder.appName("App").config("spark.driver.bindAddress", "127.0.0.1").getOrCreate()
    df = spark.read.option("header", "true").csv(
        "./dataset/unpivoted_data_10k.csv",
        inferSchema=True,
    ).limit(20)
    _, numerical_cols = utils.infer_column_types_from_schema(df.schema)

    classifier = ColumnClassifier(df[numerical_cols], hf_token=hf_token)

    results = classifier.classify_numerical_columns()

    print(json.dumps(results, indent=4))  # Output only numerical columns that need casting to datetime
