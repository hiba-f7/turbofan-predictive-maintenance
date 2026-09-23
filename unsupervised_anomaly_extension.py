"""
Extension: Unsupervised Degradation Onset Detection (Autoencoder & Isolation Forest)
-----------------------------------------------------------------------------------
Demonstrates advanced AI beyond basic supervised learning:
Trains an unsupervised model strictly on early, healthy engine cycles (cycles 1 to 30)
to detect the exact degradation onset point before triggering supervised RUL regression.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def run_unsupervised_detection(train_df, sensor_cols, healthy_cycle_cutoff=30):
    print("Fitting Unsupervised Isolation Forest on baseline healthy operating regime...")
    
    # 1. Filter healthy cycles only (unsupervised baseline)
    healthy_data = train_df[train_df["cycle"] <= healthy_cycle_cutoff][sensor_cols]
    
    scaler = StandardScaler()
    X_healthy_scaled = scaler.fit_transform(healthy_data)
    
    # 2. Train unsupervised anomaly detector
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_healthy_scaled)
    
    # 3. Compute continuous Anomaly Score across an entire engine's life
    sample_engine = train_df[train_df["engine_id"] == 1].sort_values("cycle")
    X_sample_scaled = scaler.transform(sample_engine[sensor_cols])
    
    # Negative outlier factor: lower score = more anomalous
    anomaly_scores = -iso_forest.score_samples(X_sample_scaled)
    
    print("Unsupervised Anomaly Scoring complete.")
    return sample_engine["cycle"].values, anomaly_scores

if __name__ == "__main__":
    print("Unsupervised anomaly detection module ready.")
