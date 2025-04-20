from pyspark.sql import DataFrame
import aiohttp
import asyncio
import os
import certifi
import ssl


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

    async def _classify_with_llm(self, name, dtype, values):
        ssl_context = ssl.create_default_context(cafile=certifi.where())

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

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            try:
                async with session.post(self.hf_url, json=payload, headers=headers, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("labels", ["unknown"])[0]
                    else:
                        error_text = await resp.text()
                        print(f"[ERROR] Column: '{name}' - Status: {resp.status} - Response: {error_text}")
                        return "unknown"

            except Exception as e:
                print(f"Error with LLM API request for column '{name}': {str(e)}")
                return "unknown"

    async def classify_columns(self):
        """Classify all columns in the DataFrame in parallel."""
        tasks = []
        metadata = []

        for field in self.df.schema.fields:
            name = field.name
            dtype = field.dataType.simpleString()
            values = self._get_sample_values(name)

            if self.llm_enabled and values:
                tasks.append(self._classify_with_llm(name, dtype, values))
                metadata.append((name, dtype))
            else:
                # Handle case with no values or LLM disabled
                metadata.append((name, dtype))
                tasks.append(asyncio.sleep(0, result="unknown"))  # dummy awaitable

        classified_types = await asyncio.gather(*tasks)

        results = [
            {
                "column": name,
                "spark_type": dtype,
                "classified_type": classified
            }
            for (name, dtype), classified in zip(metadata, classified_types)
        ]

        return results


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

    results = asyncio.run(classifier.classify_columns())

    for result in results:
        print(result)
