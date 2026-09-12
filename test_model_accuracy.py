import numpy as np
import pandas as pd
from similarity_engine import load_data, find_similar_players

# 1. Base Tactical Meta-Families (Discrete mapping)
BASE_TACTICAL_FAMILIES = {
    # Goalkeepers
    "Sweeper Keeper": "Modern Buildup Keeper",
    "Ball-Playing Keeper": "Modern Buildup Keeper",
    "Shot Stopper": "Traditional Stopper",
    "Traditional Stopper": "Traditional Stopper",

    # Defenders
    "Ball-Playing CB": "Possession Initiator",
    "Inverted Fullback / CB": "Possession Initiator",
    "Complete Stopper": "Possession Initiator",
    "Defensive Stopper": "Defensive Anchor",
    "Aggressive Stopper": "Defensive Anchor",
    "Aerial Dominant CB": "Defensive Anchor",

    # Midfielders
    "Holding Playmaker": "Deep Buildup Controller",
    "Deep-Lying Playmaker": "Deep Buildup Controller",
    "Deep-Lying Engine": "Deep Buildup Controller",
    "Holding Destroyer": "Transition & Ball Winner",
    "Box-to-Box Destroyer": "Transition & Ball Winner",
    "Pure Ball Winner": "Transition & Ball Winner",
    "Box-to-Box Progressor": "Transition & Ball Winner",
    "Dynamic Progressor": "Transition & Ball Winner",

    # Attackers
    "Box Poacher": "Central Finisher",
    "Complete Target Forward": "Central Finisher",
    "Power Forward": "Central Finisher",
    "Channel Runner": "Central Finisher",
    "Dynamic Striker": "Central Finisher",
    "Inside Forward": "Creative & Wide Attacker",
    "Creative Winger": "Creative & Wide Attacker",
    "Direct Winger": "Creative & Wide Attacker",
    "Creative Playmaker": "Creative & Wide Attacker",
    "Deep-Lying Link Striker": "Central Finisher"
}

# 2. Multi-Label Tagging Registry for Hybrid / Multi-Functional Profiles
# Maps players to their secondary tactical families
HYBRID_TACTICAL_REGISTRY = {
    # Attackers who link midfield buildup and create like wide playmakers
    "Joshua Zirkzee": ["Creative & Wide Attacker"],
    "Florian Wirtz": ["Central Finisher"],
    "Alexander Isak": ["Creative & Wide Attacker"],

    # Midfielders who combine progressive volume with destructive defending
    "Declan Rice": ["Deep Buildup Controller"],
    "Bruno Guimarães": ["Transition & Ball Winner"],
    "Aurélien Tchouaméni": ["Deep Buildup Controller"],

    # Defenders who carry/invert like fullbacks or sweep like high-line initiators
    "Riccardo Calafiori": ["Defensive Anchor"],
    # Extreme wide-flank crossing and carrying output
    "Alessandro Bastoni": ["Creative & Wide Attacker"]
}


def get_player_tags(player_name, raw_role):
    """
    Returns a set of all valid tactical families (Primary + any Secondary overlaps).
    """
    primary_family = BASE_TACTICAL_FAMILIES.get(raw_role, raw_role)
    secondary_families = HYBRID_TACTICAL_REGISTRY.get(player_name, [])
    return {primary_family} | set(secondary_families)


def is_tactical_compat(target_name, target_role, candidate_name, candidate_role):
    """
    Checks if there is any tactical intersection between target and candidate capabilities.
    """
    target_tags = get_player_tags(target_name, target_role)
    cand_tags = get_player_tags(candidate_name, candidate_role)
    return len(target_tags & cand_tags) > 0


def evaluate_recommender(k=3):
    df = load_data()
    total_queries = 0
    precision_scores = []
    reciprocal_ranks = []

    print(
        f"--- Running Multi-Label Tactical Compatibility Evaluation (Top-{k}) ---")

    for _, player in df.iterrows():
        name = player["player_name"]
        role = player["primary_role"]
        pos = player["position"]

        # All peers in the same position pool (excluding the player themselves)
        available_peers = df[(df["position"] == pos) & (
            df["player_name"] != name)].copy()

        # Count peers that have at least one overlapping tactical family
        target_tags = get_player_tags(name, role)
        compatible_peers_count = sum(
            len(target_tags & get_player_tags(
                r["player_name"], r["primary_role"])) > 0
            for _, r in available_peers.iterrows()
        )

        if compatible_peers_count == 0:
            continue

        eval_k = min(k, compatible_peers_count)

        results, _, _ = find_similar_players(
            df, reference_player_name=name, top_n=k)
        if results.empty:
            continue

        total_queries += 1

        # Check hits across the Top-K recommendations
        hits = 0
        hit_indices = []
        for rank_idx, (_, cand_row) in enumerate(results.head(eval_k).iterrows()):
            if is_tactical_compat(name, role, cand_row["player_name"], cand_row["primary_role"]):
                hits += 1
                hit_indices.append(rank_idx)

        precision_scores.append(hits / eval_k)

        # Mean Reciprocal Rank (MRR) based on first valid tactical match
        if len(hit_indices) > 0:
            reciprocal_ranks.append(1.0 / (hit_indices[0] + 1))
        else:
            reciprocal_ranks.append(0.0)

    avg_precision = np.mean(precision_scores) * 100.0
    mrr = np.mean(reciprocal_ranks)

    print(f"Evaluated Players    : {total_queries} / {len(df)}")
    print(
        f"Adjusted Precision@{k}: {avg_precision:.2f}% (Candidates matching primary/secondary tactical capabilities)")
    print(f"Mean Reciprocal Rank : {mrr:.3f}")

    if avg_precision >= 70.0:
        print(
            "\n[VERDICT] PASS: Archetype vectors are tightly clustered and account for hybrid profiles.")
    else:
        print("\n[VERDICT] NEEDS TUNING: Positional spaces require adjustment.")


if __name__ == "__main__":
    evaluate_recommender(k=3)
