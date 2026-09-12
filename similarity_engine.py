import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Mapping specific pitch positions to evaluation metric cohorts
POSITION_TO_COHORT = {
    "GK": "Goalkeeper",
    "CB": "Center-Back",
    "LB": "Fullback/Wingback",
    "RB": "Fullback/Wingback",
    "LWB": "Fullback/Wingback",
    "RWB": "Fullback/Wingback",
    "CDM": "Defensive-Midfield",
    "CM": "Central-Midfield",
    "CAM": "Attacking-Midfield",
    "LM": "Attacking-Midfield",
    "RM": "Attacking-Midfield",
    "LW": "Winger",
    "RW": "Winger",
    "ST": "Striker"
}

ROLE_METRICS = {
    "Goalkeeper": {
        "features": [
            "psxg_net_per90", "cross_stop_pct", "def_actions_outside_pen_per90",
            "pass_launch_pct", "pass_completion_pct", "passes_under_pressure_per90"
        ],
        "weights": [1.5, 1.2, 1.8, 1.6, 1.0, 1.0],
        "readable": ["PSxG Net/90", "Cross Stop %", "Def Outside Box", "Launch %", "Pass Cmp %", "Pressured Passes"]
    },
    "Center-Back": {
        "features": [
            "progressive_passes_per90", "progressive_pass_distance_per90", "progressive_carries_per90",
            "passes_into_final_third_per90", "pass_completion_pct", "passes_under_pressure_per90",
            "aerial_duels_won_pct", "tackles_interceptions_per90"
        ],
        "weights": [1.6, 1.2, 1.4, 1.2, 1.0, 1.4, 1.2, 1.2],
        "readable": ["Prog Passes", "Prog Dist", "Prog Carries", "Final 3rd Passes", "Pass %", "Pressured Passes", "Aerial Win %", "Tkl+Int/90"]
    },
    "Fullback/Wingback": {
        "features": [
            "progressive_carries_per90", "progressive_passes_per90", "passes_into_final_third_per90",
            "xag_per90", "tackles_interceptions_per90", "passes_under_pressure_per90",
            "take_on_success_pct", "touches_att_pen_per90"
        ],
        "weights": [1.6, 1.5, 1.4, 1.6, 1.4, 1.1, 1.4, 1.2],
        "readable": ["Prog Carries", "Prog Passes", "Final 3rd Passes", "xAG/90", "Tkl+Int/90", "Pressured Passes", "Take-on %", "Box Touches"]
    },
    "Defensive-Midfield": {
        "features": [
            "progressive_passes_per90", "passes_under_pressure_per90", "tackles_interceptions_per90",
            "pass_completion_pct", "passes_into_final_third_per90", "progressive_carries_per90",
            "aerial_duels_won_pct", "xag_per90"
        ],
        "weights": [1.8, 1.6, 2.0, 1.2, 1.4, 1.2, 1.4, 1.0],
        "readable": ["Prog Passes", "Pressured Passes", "Tkl+Int/90", "Pass %", "Final 3rd Passes", "Prog Carries", "Aerial Win %", "xAG/90"]
    },
    "Central-Midfield": {
        "features": [
            "progressive_passes_per90", "progressive_carries_per90", "passes_into_final_third_per90",
            "passes_under_pressure_per90", "pass_completion_pct", "tackles_interceptions_per90",
            "aerial_duels_won_pct", "xag_per90"
        ],
        "weights": [1.6, 1.4, 1.4, 1.4, 1.0, 1.6, 1.1, 1.4],
        "readable": ["Prog Passes", "Prog Carries", "Final 3rd Passes", "Pressured Passes", "Pass %", "Tkl+Int/90", "Aerial Win %", "xAG/90"]
    },
    "Attacking-Midfield": {
        "features": [
            "xag_per90", "progressive_passes_per90", "progressive_carries_per90",
            "passes_into_final_third_per90", "npxg_per90", "take_on_success_pct",
            "touches_att_pen_per90", "passes_under_pressure_per90"
        ],
        "weights": [1.8, 1.6, 1.5, 1.4, 1.4, 1.2, 1.4, 1.1],
        "readable": ["xAG/90", "Prog Passes", "Prog Carries", "Final 3rd Passes", "npxG/90", "Take-on %", "Box Touches", "Pressured Passes"]
    },
    "Winger": {
        "features": [
            "take_on_success_pct", "progressive_carries_per90", "touches_att_pen_per90",
            "xag_per90", "npxg_per90", "passes_under_pressure_per90", "aerial_duels_won_pct"
        ],
        "weights": [1.8, 1.6, 1.6, 1.6, 1.4, 1.0, 0.8],
        "readable": ["Take-on %", "Prog Carries", "Box Touches", "xAG/90", "npxG/90", "Pressured Passes", "Aerial Win %"]
    },
    "Striker": {
        "features": [
            "npxg_per90", "touches_att_pen_per90", "xag_per90",
            "aerial_duels_won_pct", "progressive_carries_per90", "passes_under_pressure_per90"
        ],
        "weights": [2.0, 1.6, 1.4, 1.2, 1.2, 1.0],
        "readable": ["npxG/90", "Box Touches", "xAG/90", "Aerial Win %", "Prog Carries", "Pressured Passes"]
    }
}


def load_data():
    path = os.path.join("data", "scouting_master_2024.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run python 01_fetch_data.py first.")
    return pd.read_csv(path)


def find_similar_players(df, reference_player_name, max_age=None, selected_leagues=None, strict_position=False, top_n=5):
    match = df[df["player_name"].str.lower() == reference_player_name.lower()]
    if match.empty:
        raise ValueError(
            f"Player '{reference_player_name}' not found in database.")

    player_pos = match["position"].values[0]
    cohort = POSITION_TO_COHORT.get(player_pos, "Central-Midfield")
    metrics = ROLE_METRICS[cohort]["features"]
    weights = np.array(ROLE_METRICS[cohort]["weights"])

    # Build evaluation cohort
    if strict_position:
        # Strictly the same position (e.g. only LB to LB)
        df_pool = df[df["position"] == player_pos].copy()
    else:
        # Match against the broader functional cohort (e.g., LB + LWB + RB + RWB)
        valid_positions = [pos for pos,
                           coh in POSITION_TO_COHORT.items() if coh == cohort]
        df_pool = df[df["position"].isin(valid_positions)].copy()

    # Ensure reference player is present in pool for normalization
    if reference_player_name.lower() not in df_pool["player_name"].str.lower().values:
        df_pool = pd.concat([df_pool, match], ignore_index=True)

    df_pool = df_pool.reset_index(drop=True)

    # 1. Standardize features
    scaler = StandardScaler()
    std_matrix = scaler.fit_transform(df_pool[metrics])

    # 2. Apply calibrated tactical weights
    weighted_matrix = std_matrix * np.sqrt(weights)

    # 3. Reference player vector
    ref_idx = df_pool[df_pool["player_name"].str.lower(
    ) == reference_player_name.lower()].index[0]
    query_vector = weighted_matrix[ref_idx].reshape(1, -1)

    # 4. Weighted Euclidean distance
    distances = euclidean_distances(weighted_matrix, query_vector).flatten()

    # 5. Gaussian similarity kernel
    gamma = 1.0 / (4.0 * len(metrics))
    similarity_scores = np.exp(-gamma * (distances ** 2)) * 100.0
    df_pool["similarity_score"] = np.round(similarity_scores, 2)

    # Remove self-match
    df_pool = df_pool.drop(index=ref_idx).reset_index(drop=True)

    # Apply filters
    if max_age is not None:
        df_pool = df_pool[df_pool["age"] <= max_age]

    if selected_leagues is not None and len(selected_leagues) > 0:
        df_pool = df_pool[df_pool["league"].isin(selected_leagues)]

    ranked_df = df_pool.sort_values(
        by="similarity_score", ascending=False).reset_index(drop=True)
    return ranked_df.head(top_n), metrics, cohort
