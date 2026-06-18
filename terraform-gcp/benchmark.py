#!/usr/bin/env python3
"""
LightGBM Benchmark on GCP e2-standard-8
Dataset: Credit Card Fraud Detection (284,807 transactions)
"""
import time
import json
import os
import urllib.request
import zipfile

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score
)
import lightgbm as lgb

print("=" * 60)
print("  LightGBM Benchmark - Credit Card Fraud Detection")
print("  Instance: e2-standard-8 (8 vCPU, 32GB RAM)")
print("=" * 60)

# --- Download dataset ---
DATA_DIR = os.path.expanduser("~/ml-benchmark")
CSV_PATH = os.path.join(DATA_DIR, "creditcard.csv")
os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print("\n[1/5] Downloading dataset from alternative source...")
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    urllib.request.urlretrieve(url, CSV_PATH)
    print(f"  -> Downloaded to {CSV_PATH}")
else:
    print(f"\n[1/5] Dataset already exists at {CSV_PATH}")

# --- Load data ---
print("\n[2/5] Loading data...")
t0 = time.time()
df = pd.read_csv(CSV_PATH)
load_time = time.time() - t0
print(f"  -> Loaded {len(df):,} rows, {len(df.columns)} columns in {load_time:.4f}s")
print(f"  -> Fraud ratio: {df['Class'].mean():.4%}")

# --- Prepare data ---
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  -> Train: {len(X_train):,}, Test: {len(X_test):,}")

# --- Train LightGBM ---
print("\n[3/5] Training LightGBM...")
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": 577,  # ratio of negatives to positives (~99.83% vs 0.17%)
    "verbose": -1,
    "num_threads": 8,
}

t0 = time.time()
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100),
    ],
)
train_time = time.time() - t0
print(f"  -> Training complete in {train_time:.4f}s")
print(f"  -> Best iteration: {model.best_iteration}")

# --- Evaluate ---
print("\n[4/5] Evaluating model...")
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

# --- Inference latency ---
print("\n[5/5] Measuring inference latency...")
# Single row
iterations = 1000
t0 = time.time()
for _ in range(iterations):
    model.predict(X_test.iloc[:1])
latency_1row = (time.time() - t0) / iterations

# Batch of 1000 rows
t0 = time.time()
for _ in range(100):
    model.predict(X_test.iloc[:1000])
throughput_1000 = (time.time() - t0) / 100

# --- Results ---
results = {
    "instance_type": "e2-standard-8",
    "vcpu": 8,
    "ram_gb": 32,
    "dataset": "Credit Card Fraud Detection",
    "dataset_rows": len(df),
    "load_time_sec": round(load_time, 4),
    "train_time_sec": round(train_time, 4),
    "best_iteration": model.best_iteration,
    "auc_roc": round(auc, 6),
    "accuracy": round(acc, 6),
    "f1_score": round(f1, 6),
    "precision": round(prec, 6),
    "recall": round(rec, 6),
    "inference_latency_1row_ms": round(latency_1row * 1000, 4),
    "inference_throughput_1000rows_ms": round(throughput_1000 * 1000, 4),
}

print("\n" + "=" * 60)
print("  BENCHMARK RESULTS")
print("=" * 60)
for k, v in results.items():
    print(f"  {k:.<40s} {v}")

# Save results
result_path = os.path.join(DATA_DIR, "benchmark_result.json")
with open(result_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Results saved to {result_path}")
print("=" * 60)
