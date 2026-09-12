import os
import sys

# 1. Environment variables to fix Windows networking and Python worker mismatches
os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['PATH'] += os.pathsep + 'C:\\hadoop\\bin'
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

# FORCE Spark to use the active Conda environment for background workers
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import joblib
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, pandas_udf
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

# 2. Initialize Spark Session
spark = SparkSession.builder \
    .appName("SentinelAI-Kaggle-Inference") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("Error")
print("Spark Session successfully started. Loading ML model...")

# 3. Load and broadcast the trained model
model = joblib.load("fraud_model.joblib")
broadcast_model = spark.sparkContext.broadcast(model) # sending copy of trained model to all of the cpu core 

# 4. Define the exact feature names expected by your Kaggle model
FEATURE_NAMES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'] # create column name v0-v29

# 5. Construct the PySpark schema for incoming Kaggle payloads
schema_fields = [StructField(feature, DoubleType(), True) for feature in FEATURE_NAMES] # data type of v0-20
schema_fields.append(StructField("Class", IntegerType(), True)) # adding one more column class 
schema = StructType(schema_fields) #The computer finalizes these 31 rules into a strict blueprint variable named schema

# 6. Connect to Kafka stream
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "live-transactions") \
    .option("startingOffsets", "latest") \
    .load()

# 7. Parse JSON payload into structured DataFrame
parsed_stream = raw_stream.selectExpr("CAST(value AS STRING) as json_payload") \
    .select(from_json(col("json_payload"), schema).alias("data")) \
    .select("data.*")

# 8. Vectorized Pandas UDF for multi-column batch inference
@pandas_udf(DoubleType())
def predict_fraud_udf(*cols: pd.Series) -> pd.Series:
    features = pd.concat(cols, axis=1)
    features.columns = FEATURE_NAMES
    probabilities = broadcast_model.value.predict_proba(features)[:, 1]
    return pd.Series(probabilities)

# 9. Score live stream by passing all feature columns to the UDF
feature_columns = [col(f) for f in FEATURE_NAMES]
scored_stream = parsed_stream.withColumn(
    "fraud_probability",
    predict_fraud_udf(*feature_columns)
)

# 10. Filter for high-risk anomalies (probability > 70%)
flagged_anomalies = scored_stream.filter(col("fraud_probability") > 0.70) \
    .select("Time", "Amount", "fraud_probability", "Class")

# 11. Custom function to write each micro-batch to a single text file
def save_to_text_file(df, epoch_id):
    pandas_df = df.toPandas()
    
    if not pandas_df.empty:
        with open("flagged_alerts.txt", "a") as file:
            for index, row in pandas_df.iterrows():
                alert_msg = f"FRAUD ALERT | Time ID: {row['Time']} | Amount: ${row['Amount']:.2f} | AI Score: {row['fraud_probability']:.4f}\n"
                file.write(alert_msg)
                print(alert_msg.strip())

# 12. Route the flagged stream through the custom text-writer function
query = flagged_anomalies.writeStream \
    .outputMode("append") \
    .foreachBatch(save_to_text_file) \
    .trigger(processingTime="2 seconds") \
    .start()

print("Real-time ML scoring is active. Logging live anomalies to 'flagged_alerts.txt'...")
query.awaitTermination()