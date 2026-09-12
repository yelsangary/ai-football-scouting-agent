import os
import matplotlib.pyplot as plt
from mplsoccer import PyPizza
from similarity_engine import POSITION_TO_COHORT, ROLE_METRICS


def calculate_percentiles(df, player_name, cohort, metrics):
    player_row = df[df["player_name"].str.lower() == player_name.lower()]
    if player_row.empty:
        return [50.0] * len(metrics)

    valid_positions = [pos for pos,
                       coh in POSITION_TO_COHORT.items() if coh == cohort]
    peer_pool = df[df["position"].isin(valid_positions)]

    percentiles = []
    for col in metrics:
        pctile = (peer_pool[col] < player_row[col].values[0]).mean() * 100.0
        percentiles.append(round(pctile, 1))
    return percentiles


def plot_scouting_pizza(df, candidate_name, benchmark_name, save_filename=None):
    candidate_row = df[df["player_name"].str.lower() == candidate_name.lower()]
    pos = candidate_row["position"].values[0]
    cohort = POSITION_TO_COHORT.get(pos, "Central-Midfield")

    metrics = ROLE_METRICS[cohort]["features"]
    readable_params = ROLE_METRICS[cohort]["readable"]

    candidate_pctiles = calculate_percentiles(
        df, candidate_name, cohort, metrics)
    benchmark_pctiles = calculate_percentiles(
        df, benchmark_name, cohort, metrics)

    os.makedirs("outputs", exist_ok=True)
    bg_color = "#121212"
    slice_color = "#1A78CF"
    compare_color = "#22C55E"
    text_color = "#FFFFFF"

    baker = PyPizza(
        params=readable_params,
        background_color=bg_color,
        straight_line_color="#333333",
        straight_line_lw=1.2,
        last_circle_lw=1.2,
        last_circle_color="#444444",
        other_circle_lw=0.6,
        other_circle_color="#2b2b2b",
        inner_circle_size=15
    )

    fig, ax = baker.make_pizza(
        candidate_pctiles,
        compare_values=benchmark_pctiles,
        figsize=(9, 9.5),
        param_location=110,
        slice_colors=[slice_color] * len(metrics),
        compare_colors=[compare_color] * len(metrics),
        color_blank_space="same",
        blank_alpha=0.12,
        kwargs_params=dict(color="#F2F2F2", fontsize=10.5,
                           weight="bold", va="center"),
        kwargs_values=dict(color="#FFFFFF", fontsize=9.5, zorder=4, bbox=dict(
            edgecolor="#121212", facecolor=slice_color, boxstyle="round,pad=0.25", lw=0.8)),
        kwargs_compare_values=dict(color="#121212", fontsize=9.5, weight="bold", zorder=4, bbox=dict(
            edgecolor="#121212", facecolor=compare_color, boxstyle="round,pad=0.25", lw=0.8))
    )

    fig.text(0.515, 0.98, f"{candidate_name} vs {benchmark_name}",
             size=16, ha="center", color=text_color, weight="bold")
    fig.text(0.515, 0.95, f"Position: {pos} ({cohort}) | Blue: {candidate_name} | Green: {benchmark_name}",
             size=10.5, ha="center", color="#A0A0A0")

    if save_filename is None:
        save_filename = f"{candidate_name.replace(' ', '_').lower()}_vs_bench.png"

    save_path = os.path.join("outputs", save_filename)
    fig.savefig(save_path, dpi=250, facecolor=bg_color, bbox_inches="tight")
    plt.close()
    return save_path
