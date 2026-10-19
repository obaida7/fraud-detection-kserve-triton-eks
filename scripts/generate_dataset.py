#!/usr/bin/env python3
"""
generate_dataset.py
Generates a realistic, standardized banking fraud detection dataset.
Features match standard banking transaction telemetry:
- Time: Seconds elapsed since baseline
- Amount: Transaction amount (log-normal distribution)
- V1-V28: PCA-transformed behavioral, velocity, geo-risk, and device embeddings
- Class: 0 = Legitimate, 1 = Fraudulent (~1.5% class imbalance)
"""

import os
import argparse
import numpy as np
import pandas as pd

def generate_fraud_data(num_samples: int = 25000, fraud_ratio: float = 0.015, random_seed: int = 42):
    np.random.seed(random_seed)
    num_fraud = int(num_samples * fraud_ratio)
    num_legit = num_samples - num_fraud
    
    print(f"Generating {num_samples} banking transactions ({num_legit} legitimate, {num_fraud} fraudulent)...")
    
    # 1. Generate Timestamps (seconds over 48 hours)
    time_legit = np.sort(np.random.uniform(0, 172800, num_legit))
    time_fraud = np.sort(np.random.uniform(0, 172800, num_fraud))
    
    # 2. Generate Amounts
    # Legitimate transactions: typical retail purchases (mean ~$80, std log scale)
    amount_legit = np.round(np.random.lognormal(mean=3.8, sigma=1.2, size=num_legit), 2)
    # Fraud transactions: mix of micro-probing ($1-$5) and large bust-out thefts ($500 - $3500)
    fraud_micro = np.round(np.random.uniform(1.0, 5.0, size=num_fraud // 3), 2)
    fraud_large = np.round(np.random.lognormal(mean=6.5, sigma=0.9, size=num_fraud - len(fraud_micro)), 2)
    amount_fraud = np.concatenate([fraud_micro, fraud_large])
    np.random.shuffle(amount_fraud)
    
    # 3. Generate V1 - V28 Features (PCA Latent Vectors)
    # Legitimate: standard normal distributions around 0
    v_legit = np.random.normal(loc=0.0, scale=1.0, size=(num_legit, 28))
    
    # Fraudulent: shifted distributions on key fraud-predictive indicators (V4, V10, V12, V14, V17)
    v_fraud = np.random.normal(loc=0.0, scale=1.2, size=(num_fraud, 28))
    v_fraud[:, 3] += np.random.normal(loc=3.5, scale=1.0, size=num_fraud)   # V4: velocity spike
    v_fraud[:, 9] -= np.random.normal(loc=4.0, scale=1.0, size=num_fraud)   # V10: high-risk merchant
    v_fraud[:, 11] -= np.random.normal(loc=5.0, scale=1.0, size=num_fraud)  # V12: device mismatch
    v_fraud[:, 13] -= np.random.normal(loc=6.0, scale=1.1, size=num_fraud)  # V14: geo-distance anomaly
    v_fraud[:, 16] -= np.random.normal(loc=4.5, scale=1.0, size=num_fraud)  # V17: cross-border surge
    
    # Assemble DataFrames
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    
    data_legit = np.column_stack([time_legit, v_legit, amount_legit, np.zeros(num_legit)])
    data_fraud = np.column_stack([time_fraud, v_fraud, amount_fraud, np.ones(num_fraud)])
    
    all_data = np.vstack([data_legit, data_fraud])
    df = pd.DataFrame(all_data, columns=cols)
    
    # Sort chronologically by Time
    df = df.sort_values(by="Time").reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate Banking Fraud Transaction Dataset")
    parser.add_argument("--samples", type=int, default=25000, help="Total number of transaction records")
    parser.add_argument("--fraud_ratio", type=float, default=0.015, help="Proportion of fraudulent transactions")
    parser.add_argument("--output", type=str, default="data/fraud_detection.csv", help="Output path for CSV")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = generate_fraud_data(num_samples=args.samples, fraud_ratio=args.fraud_ratio)
    df.to_csv(args.output, index=False)
    
    print(f"Dataset successfully created at: {args.output}")
    print(f"Total Rows: {len(df)}, Fraud Count: {df['Class'].sum()} ({df['Class'].mean():.2%})")

if __name__ == "__main__":
    main()
