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

    def _build_model_context(self, name, dtype, values):
        # Build model context for each column to be sent to the model for classification
        return f"""
            Task: column_type_classification. 
            Guidance:
            - If the datatype is 'string', classify the column as 'categorical'.
            - If the datatype is numerical and the values appear very large (e.g., 10-digit integers), consider whether they might be Unix timestamps. Classify such columns as 'date' or 'datetime', 'numeric' otherwise.
            Column Metadata:
            {{
            "column name": "{name}",
            "spark datatype": "{dtype}",
            "sample of values": [{", ".join(f'"{str(v)}"' for v in values[:3])}]
            }}
        """

    def _classify_with_llm(self, name, dtype, values):
        input_text = self._build_model_context(name, dtype, values)

        payload = {
            "inputs": input_text,
            "parameters": {
                "candidate_labels": ["categorical", "numerical", "date", "datetime", "unknown"]
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

    def classify_columns(self) -> dict:
        numerical_columns = []
        categorical_columns = []
        to_cast = []

        for field in self.df.schema.fields:
            name = field.name
            dtype = field.dataType.simpleString()
            values = self._get_sample_values(name)

            if self.llm_enabled and values:
                classified_type = self._classify_with_llm(name, dtype, values)
            else:
                # Handle case with no values or LLM disabled
                classified_type = "unknown"

            # Classify based on the LLM's response
            if classified_type == "categorical":
                categorical_columns.append(name)
            elif classified_type == "date" or classified_type == "datetime":
                to_cast.append((name, "datetime" if classified_type == "datetime" else "date"))
            else:
                numerical_columns.append(name)

        return {
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "to_cast": to_cast
        }


if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from dotenv import load_dotenv

    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")

    spark = SparkSession.builder.appName("App").config("spark.driver.bindAddress", "127.0.0.1").getOrCreate()
    df = spark.read.option("header", "true").csv(
        "./dataset/unpivoted_data_10k.csv",
        inferSchema=True,
    ).limit(20)

    classifier = ColumnClassifier(df, hf_token=hf_token)

    results = classifier.classify_columns()

    print(json.dumps(results, indent=4))  # Print the results in a formatted JSON
