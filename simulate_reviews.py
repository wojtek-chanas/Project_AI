"""
Review Stream Simulator
=======================
Simulates real-time ingestion of customer reviews by sending randomly
selected reviews from the Amazon dataset to the /predict endpoint
at random intervals between 1 and 10 seconds.

Usage:
    python simulate_reviews.py
"""

import requests
import pandas as pd
import time
import random

API_URL = "http://localhost:8000"
DATASET_PATH = 'AmazonProductReviews.csv'
MIN_INTERVAL = 1   # seconds
MAX_INTERVAL = 10  # seconds

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH, low_memory=False)
reviews = df['reviews.text'].dropna().tolist()
print(f"Loaded {len(reviews):,} reviews. Starting simulation...\n")

count = 0

while True:
    # Pick a random review
    text = random.choice(reviews)

    # Send to API
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": str(text)}
        )
        result = response.json()
        count += 1
        print(f"[{count:>4}] {result['sentiment'].upper():8} ({result['confidence']:.0%}) | {text[:60]}...")

    except requests.ConnectionError:
        print("ERROR: API not reachable. Is uvicorn running?")
        break

    # Random pause before next review
    interval = random.uniform(MIN_INTERVAL, MAX_INTERVAL)
    time.sleep(interval)
