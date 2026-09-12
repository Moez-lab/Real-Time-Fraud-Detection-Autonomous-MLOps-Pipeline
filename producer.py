import json
import time
import os
import random
import pandas as pd
from confluent_kafka import Producer

# 1. Connect to local Kafka broker
conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed: {err}")
    else:
        print(f"Sent record to Kafka | Offset: {msg.offset()}")

CSV_FILE = "creditcard.csv"

print("Starting Sentinel AI Kaggle stream... Press Ctrl+C to stop.")

try:
    if os.path.exists(CSV_FILE):
        print(f"Reading records from {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        
        for _, row in df.iterrows():
            payload = row.to_dict()
            json_payload = json.dumps(payload).encode('utf-8')
            producer.produce('live-transactions', value=json_payload, callback=delivery_report)
            producer.poll(0)
            time.sleep(0.01)  # Stream a transaction every 100ms
    else:
        print("creditcard.csv not found locally. Simulating Kaggle schema...")
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        while True:
            is_fraud = 1 if random.random() < 0.05 else 0
            simulated_row = {f'V{i}': round(random.gauss(0, 1), 4) for i in range(1, 29)}
            simulated_row['Time'] = round(time.time(), 2)
            simulated_row['Amount'] = round(random.uniform(500.0, 3000.0), 2) if is_fraud else round(random.uniform(5.0, 150.0), 2)
            simulated_row['Class'] = is_fraud
            
            json_payload = json.dumps(simulated_row).encode('utf-8')
            producer.produce('live-transactions', value=json_payload, callback=delivery_report)
            producer.poll(0)
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping data stream...")

producer.flush()