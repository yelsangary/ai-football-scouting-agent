import streamlit as st
import pandas as pd
import os

from similarity_engine import find_similar_players, load_data, POSITION_TO_COHORT, ROLE_METRICS
from visualize_scout import plot_scouting_pizza
from ai_scout_dossier import extract_scout_payload, generate_dossier

st.set_page_config(
    page_title="AI Tactical Recruitment Engine",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161B22; padding: 12px; border-radius: 8px; border: 1px solid #30363D; }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Elite Football Recruitment & Tactical Archetype Engine")
st.caption(
    "Granular Positional Profiling, League Filtering, and AI Executive Scouting Dossiers")

try:
    df = load_data()
except Exception as e:
    st.error(
        f"Error loading database: {e}. Run 'python 01_fetch_data.py' first.")
    st.stop()

# -------------------------------------------------------------
# 1. TOP BAR: DIRECT PLAYER SEARCH OR BROWSER
# -------------------------------------------------------------
search_col1, search_col2 = st.columns([3, 1])

all_players_list = sorted(df["player_name"].unique().tolist())
with search_col1:
    search_query = st.selectbox(
        "🔍 Instant Player Search (Type any player name to scout):",
        options=[""] + all_players_list,
        index=0,
        help="Start typing to immediately pull up any specific player in the scouting database."
    )

with search_col2:
    target_club = st.selectbox(
        "Scouting on Behalf of Club:",
        ["Manchester United", "Arsenal", "Manchester City",
            "Liverpool", "Real Madrid", "Bayern Munich"]
    )

st.markdown("---")

# -------------------------------------------------------------
# 2. SIDEBAR PARAMETERS (GRANULAR POSITIONS & LEAGUES)
# -------------------------------------------------------------
st.sidebar.header("🎯 Tactical Scouting Parameters")

# Position categories with exact football pitch codes
POSITION_GROUPS = {
    "Goalkeepers": ["GK"],
    "Defenders": ["CB", "LB", "RB", "LWB", "RWB"],
    "Midfielders": ["CDM", "CM", "CAM", "LM", "RM"],
    "Attackers": ["ST", "LW", "RW"]
}

pos_category = st.sidebar.selectbox(
    "Positional Unit", list(POSITION_GROUPS.keys()))
available_exact_positions = POSITION_GROUPS[pos_category]
selected_exact_pos = st.sidebar.selectbox(
    "Specific Pitch Position", available_exact_positions)

# Dynamic League Multi-Select from Dataset
all_available_leagues = sorted(df["league"].unique().tolist())
selected_leagues = st.sidebar.multiselect(
    "Eligible Scouting Leagues",
    options=all_available_leagues,
    default=all_available_leagues,
    help="Filter prospective candidates by competition."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Prospect Filters")
max_age = st.sidebar.slider(
    "Maximum Age Cap", min_value=17, max_value=35, value=25)
top_candidates_count = st.sidebar.slider(
    "Number of Candidates", min_value=3, max_value=8, value=5)
strict_pos_match = st.sidebar.checkbox(
    "Strict Position Only (e.g. Only LB, exclude LWB)", value=False)

# -------------------------------------------------------------
# 3. RESOLVE BENCHMARK PLAYER (Search Bar vs Positional Picker)
# -------------------------------------------------------------
if search_query != "":
    benchmark_player = search_query
    player_row = df[df["player_name"] == benchmark_player].iloc[0]
    st.info(
        f"📍 **Target Reference:** {benchmark_player} | Position: **{player_row['position']}** | Role: **{player_row['primary_role']}** | Club: **{player_row['team']}** ({player_row['league']})")
else:
    # Filter benchmark player list based on the chosen exact position
    pos_players = df[df["position"] ==
                     selected_exact_pos]["player_name"].sort_values().tolist()
    if not pos_players:
        st.warning(
            f"No benchmark player currently cataloged under '{selected_exact_pos}'. Please pick another position.")
        st.stop()
    benchmark_player = st.sidebar.selectbox(
        "Benchmark Player to Replace / Replicate", pos_players)

# -------------------------------------------------------------
# 4. EXECUTE SEARCH ENGINE
# -------------------------------------------------------------
results, active_metrics, cohort_name = find_similar_players(
    df,
    reference_player_name=benchmark_player,
    max_age=max_age,
    selected_leagues=selected_leagues,
    strict_position=strict_pos_match,
    top_n=top_candidates_count
)

st.subheader(
    f"Top Matches for {benchmark_player} ({cohort_name} Cohort, Age ≤ {max_age})")

if results.empty:
    st.warning(
        "No prospective candidates match the current age and league criteria. Expand filters in sidebar.")
else:
    cols_to_show = ["player_name", "team", "league", "position",
                    "primary_role", "age", "similarity_score"] + active_metrics[:3]
    display_df = results[cols_to_show].rename(columns={
        "player_name": "Player",
        "team": "Club",
        "league": "League",
        "position": "Pos",
        "primary_role": "Tactical Archetype",
        "age": "Age",
        "similarity_score": "Similarity (%)"
    })
    st.dataframe(display_df, width="stretch", hide_index=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # 5. DETAILED INSPECTION & AI DOSSIER
    # -------------------------------------------------------------
    st.subheader("📊 Candidate Tactical Evaluation")

    col_select, col_score = st.columns([3, 1])
    with col_select:
        selected_candidate = st.selectbox(
            "Select candidate to inspect in detail:",
            results["player_name"].tolist()
        )

    cand_row = results[results["player_name"] == selected_candidate].iloc[0]
    with col_score:
        st.metric("Similarity Index", f"{cand_row['similarity_score']}%")

    col_chart, col_dossier = st.columns([1.1, 1.2])

    with col_chart:
        st.markdown(
            f"**Tactical Percentile Comparison vs {benchmark_player}**")
        chart_filename = f"{selected_candidate.replace(' ', '_').lower()}_vs_bench.png"
        chart_path = plot_scouting_pizza(
            df,
            candidate_name=selected_candidate,
            benchmark_name=benchmark_player,
            save_filename=chart_filename
        )
        st.image(chart_path, width="stretch")

    with col_dossier:
        st.markdown(f"**Executive Scouting Briefing ({target_club})**")
        if st.button("⚡ Generate AI Scouting Dossier"):
            with st.spinner("Analyzing tactical metrics and synthesizing briefing..."):
                payload = extract_scout_payload(
                    df,
                    candidate_name=selected_candidate,
                    benchmark_name=benchmark_player,
                    similarity_score=cand_row["similarity_score"]
                )
                dossier_text = generate_dossier(
                    payload, target_club=target_club)
                st.text_area("Recruitment Memo", dossier_text, height=480)
        else:
            st.info(
                "Click 'Generate AI Scouting Dossier' to produce an executive recruitment memo.")
