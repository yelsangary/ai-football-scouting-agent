import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

BPD_METRICS = [
    "progressive_passes_per90",
    "progressive_pass_distance_per90",
    "progressive_carries_per90",
    "passes_into_final_third_per90",
    "pass_completion_pct_medium",
    "pass_completion_pct_long",
    "passes_under_pressure_per90",
    "aerial_duels_won_pct"
]

def engineer_features():
    raw_path = os.path.join("data", "scouting_defenders_2024.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Ensure Stage 1 ran successfully.")

    df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset with {len(df)} players.")

    MIN_MINUTES = 900
    df_filtered = df[df["minutes_played"] >= MIN_MINUTES].copy().reset_index(drop=True)
    print(f"Filtered to {len(df_filtered)} players meeting >= {MIN_MINUTES} minutes.")

    df_percentile = df_filtered.copy()
    for col in BPD_METRICS:
        percentile_col = f"{col}_percentile"
        df_percentile[percentile_col] = (df_filtered[col].rank(pct=True) *100).round(1)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_filtered[BPD_METRICS])
    df_scaled = pd.DataFrame(scaled_values, columns=[f"col_scaled" for col in BPD_METRICS])
    df_final = pd.concat([df_percentile, df_scaled], axis=1)

    processed_path = os.path.join("data", "processed_bpd_scouting.csv")
    df_final.to_csv(processed_path, index=False)

    print("\n[SUCCESS] Stage 2 Feature Engineering Complete.")
    print(f"- Processed scouting table saved to: {processed_path}")
    print(f"- Total qualified candidates: {len(df_final)}")

    l_martinez = df_final[df_final["player_name"] == "Lisandro Martinez"]
    if not l_martinez.empty:
        print("\nReference Player Check (Lisandro Martínez - Percentiles):")
        for col in BPD_METRICS:
            val = l_martinez[f"{col}_pctile"].values[0]
            raw = l_martinez[col].values[0]
            print(f"  * {col}: {raw} ({val}th percentile)")

if __name__ == "__main__":
    engineer_features()