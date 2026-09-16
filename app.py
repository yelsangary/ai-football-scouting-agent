import streamlit as st
import pandas as pd
import os

# Module imports
from similarity_engine import find_similar_players, load_data, POSITION_TO_COHORT, ROLE_METRICS
from visualize_scout import plot_scouting_pizza
from ai_scout_dossier import extract_scout_payload, generate_dossier
from portfolio_optimizer import load_financial_market_data, solve_transfer_portfolio
from ai_boardroom_pitch import generate_boardroom_pitch

st.set_page_config(
    page_title="Football Intelligence & Recruitment Hub",
    page_icon="⚽",
    layout="wide"
)

# Dark theme styling
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161B22; padding: 14px; border-radius: 8px; border: 1px solid #30363D; }
    div[data-baseweb="tab-list"] { gap: 16px; }
    button[data-baseweb="tab"] { font-size: 16px; font-weight: 600; padding: 10px 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Elite Football Intelligence & Recruitment Hub")
st.caption("Integrated Vector Similarity Scouting, Financial Re-investment Optimization, and Boardroom Dossiers")

# Global Club Selection
target_club = st.sidebar.selectbox(
    "🏛️ Operating Club / Recruitment Bureau:",
    ["Manchester United", "Arsenal", "Manchester City", "Liverpool", "Real Madrid", "Bayern Munich"]
)

# Create the two primary operating tabs
tab_twin, tab_optimizer = st.tabs([
    "🎯 Tab 1: Tactical Twin Scouting",
    "💼 Tab 2: Moneyball Portfolio Optimizer"
])

# ==============================================================================
# TAB 1: TACTICAL TWIN SCOUTING
# ==============================================================================
with tab_twin:
    try:
        df_scout = load_data()
    except Exception as e:
        st.error(f"Error loading scouting data: {e}")
        st.stop()

    search_col1, search_col2 = st.columns([3, 1])
    all_players_list = sorted(df_scout["player_name"].unique().tolist())

    with search_col1:
        search_query = st.selectbox(
            "🔍 Instant Player Search (Type to scout any player in the system):",
            options=[""] + all_players_list,
            index=0,
            key="twin_search"
        )

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Twin Prospect Filters")
    
    POSITION_GROUPS = {
        "Goalkeepers": ["GK"],
        "Defenders": ["CB", "LB", "RB", "LWB", "RWB"],
        "Midfielders": ["CDM", "CM", "CAM", "LM", "RM"],
        "Attackers": ["ST", "LW", "RW"]
    }
    
    pos_category = st.sidebar.selectbox("Positional Unit", list(POSITION_GROUPS.keys()), key="twin_pos_cat")
    available_exact_positions = POSITION_GROUPS[pos_category]
    selected_exact_pos = st.sidebar.selectbox("Pitch Position", available_exact_positions, key="twin_pos_exact")

    all_leagues = sorted(df_scout["league"].unique().tolist())
    selected_leagues = st.sidebar.multiselect(
        "Eligible Scouting Leagues",
        options=all_leagues,
        default=all_leagues,
        key="twin_leagues"
    )

    max_age = st.sidebar.slider("Maximum Age Cap", 17, 35, 25, key="twin_age")
    top_k = st.sidebar.slider("Number of Candidates", 3, 8, 5, key="twin_k")
    strict_pos = st.sidebar.checkbox("Strict Position Only (e.g., LB only, exclude LWB)", value=False, key="twin_strict")

    # Resolve Reference Player
    if search_query != "":
        benchmark_player = search_query
        ref_row = df_scout[df_scout["player_name"] == benchmark_player].iloc[0]
        st.info(f"📍 **Target Benchmark:** {benchmark_player} | Position: **{ref_row['position']}** | Archetype: **{ref_row['primary_role']}** | Club: **{ref_row['team']}** ({ref_row['league']})")
    else:
        filtered_players = df_scout[df_scout["position"] == selected_exact_pos]["player_name"].sort_values().tolist()
        if not filtered_players:
            st.warning(f"No benchmark player cataloged under '{selected_exact_pos}'.")
            st.stop()
        benchmark_player = st.sidebar.selectbox("Benchmark Player to Replicate", filtered_players, key="twin_bench_select")

    # Run Similarity Engine
    results, active_metrics, cohort_name = find_similar_players(
        df_scout,
        reference_player_name=benchmark_player,
        max_age=max_age,
        selected_leagues=selected_leagues,
        strict_position=strict_pos,
        top_n=top_k
    )

    st.subheader(f"Top Statistical Twins for {benchmark_player} ({cohort_name} Cohort)")

    if results.empty:
        st.warning("No prospective candidates match the criteria. Adjust the filters in the sidebar.")
    else:
        cols_to_show = ["player_name", "team", "league", "position", "primary_role", "age", "similarity_score"] + active_metrics[:3]
        display_df = results[cols_to_show].rename(columns={
            "player_name": "Player",
            "team": "Club",
            "league": "League",
            "position": "Pos",
            "primary_role": "Tactical Role",
            "age": "Age",
            "similarity_score": "Similarity (%)"
        })
        st.dataframe(display_df, width="stretch", hide_index=True)

        st.markdown("---")
        st.subheader("📊 Candidate Deep-Dive & Tactical Pizza Chart")

        c_select, c_score = st.columns([3, 1])
        with c_select:
            selected_cand = st.selectbox("Select candidate to benchmark:", results["player_name"].tolist(), key="twin_cand_pick")
        cand_row = results[results["player_name"] == selected_cand].iloc[0]
        with c_score:
            st.metric("Similarity Index", f"{cand_row['similarity_score']}%")

        col_radar, col_memo = st.columns([1.1, 1.2])
        with col_radar:
            chart_file = f"{selected_cand.replace(' ', '_').lower()}_vs_bench.png"
            chart_path = plot_scouting_pizza(
                df_scout,
                candidate_name=selected_cand,
                benchmark_name=benchmark_player,
                save_filename=chart_file
            )
            st.image(chart_path, width="stretch")

        with col_memo:
            st.markdown(f"**Executive Scouting Dossier ({target_club})**")
            if st.button("⚡ Generate AI Scouting Dossier", key="twin_btn_dossier"):
                with st.spinner("Synthesizing metrics and tactical evaluation..."):
                    payload = extract_scout_payload(
                        df_scout,
                        candidate_name=selected_cand,
                        benchmark_name=benchmark_player,
                        similarity_score=cand_row["similarity_score"]
                    )
                    briefing = generate_dossier(payload, target_club=target_club)
                    st.text_area("Recruitment Memo", briefing, height=480)
            else:
                st.info("Click to produce an executive recruitment memo powered by Gemini.")

# ==============================================================================
# TAB 2: MONEYBALL PORTFOLIO OPTIMIZER
# ==============================================================================
with tab_optimizer:
    st.subheader("💼 Capital Re-allocation & Multi-Asset Portfolio Optimizer")
    st.markdown("""
        When selling a star asset, spending the windfall on a single replacement introduces extreme operational risk.
        This module runs **Mixed-Integer Linear Programming (MILP)** to identify the optimal basket of signings that collectively
        replaces your lost tactical workload while preserving capital and protecting wage structure.
    """)

    try:
        df_fin = load_financial_market_data()
    except Exception as e:
        st.error(f"Error loading financial database: {e}. Run 'python 02_build_market_db.py' first.")
        st.stop()

    st.markdown("---")
    
    # Financial Scenario Setup
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        fee_budget = st.slider("💰 Transfer Fee Budget Available (€M)", min_value=30.0, max_value=250.0, value=90.0, step=5.0)
    with opt_col2:
        wage_ceiling = st.slider("📉 Maximum Weekly Wage Budget (£k/week)", min_value=50.0, max_value=400.0, value=170.0, step=5.0)
    with opt_col3:
        target_signings = st.radio("👥 Target Signing Count (Basket Size)", [2, 3], horizontal=True)

    # Tactical Minimum Floor Setup
    st.markdown("##### 🎯 Minimum Tactical Workload Requirements (Combined Portfolio Floor)")
    tact_col1, tact_col2, tact_col3, tact_col4 = st.columns(4)
    with tact_col1:
        min_prog_passes = st.slider("Min Combined Progressive Passes/90", 6.0, 16.0, 11.0, 0.5)
    with tact_col2:
        min_def_actions = st.slider("Min Combined Defensive Actions/90", 4.0, 14.0, 7.0, 0.5)
    with tact_col3:
        min_creation = st.slider("Min Combined xG+xAG/90", 0.20, 1.50, 0.40, 0.05)
    with tact_col4:
        max_avg_age = st.slider("Max Average Portfolio Age", 20.0, 28.0, 23.5, 0.5)

    scenario_params = {
        "scenario_title": f"Reinvestment of {fee_budget}M Capital Windfall",
        "fee_budget_m": fee_budget,
        "wage_ceiling_k_pw": wage_ceiling
    }

    if st.button("🚀 Run Portfolio Optimization Solver", type="primary"):
        with st.spinner("Formulating constraint matrix and solving 0-1 MILP..."):
            basket, status_msg = solve_transfer_portfolio(
                df_fin,
                fee_budget_m=fee_budget,
                wage_ceiling_k_pw=wage_ceiling,
                target_signings_count=target_signings,
                max_avg_age=max_avg_age,
                min_prog_passes=min_prog_passes,
                min_defensive_actions=min_def_actions,
                min_creation_threat=min_creation
            )

        if basket is not None:
            st.success(f"Optimal Solution Identified ({status_msg})")
            
            # Key Metrics Display
            spent_fee = basket["market_value_eur_m"].sum()
            spent_wages = basket["wage_gbp_k_pw"].sum()
            comb_prog = basket["progressive_passes_per90"].sum()
            comb_def = basket["tackles_interceptions_per90"].sum()
            portfolio_age = basket["age"].mean()

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Capital Outlay", f"€{spent_fee:.1f}M", delta=f"€{fee_budget - spent_fee:.1f}M Saved", delta_color="normal")
            m2.metric("Committed Wages", f"£{spent_wages:.1f}k/pw", delta=f"£{wage_ceiling - spent_wages:.1f}k Saved", delta_color="normal")
            m3.metric("Combined Prog Passes", f"{comb_prog:.1f}/90", delta=f"+{comb_prog - min_prog_passes:.1f}")
            m4.metric("Combined Def Actions", f"{comb_def:.1f}/90", delta=f"+{comb_def - min_def_actions:.1f}")
            m5.metric("Average Squad Age", f"{portfolio_age:.1f} yrs", delta=f"{max_avg_age - portfolio_age:.1f} under cap")

            st.markdown("---")
            st.markdown("##### 📋 Recommended Target Acquisition Basket")
            
            display_basket = basket[[
                "player_name", "position", "team", "league", "age",
                "market_value_eur_m", "wage_gbp_k_pw", "progressive_passes_per90",
                "tackles_interceptions_per90", "resale_potential_multiplier"
            ]].rename(columns={
                "player_name": "Target",
                "position": "Pos",
                "team": "Club",
                "league": "League",
                "age": "Age",
                "market_value_eur_m": "Fee (€M)",
                "wage_gbp_k_pw": "Wage (£k/pw)",
                "progressive_passes_per90": "Prog Pass/90",
                "tackles_interceptions_per90": "Tkl+Int/90",
                "resale_potential_multiplier": "Resale Upside"
            })
            st.dataframe(display_basket, width="stretch", hide_index=True)

            st.markdown("---")
            st.markdown(f"##### 🏛️ Boardroom Investment Memo ({target_club})")
            
            with st.spinner("Generating executive investment memo with Gemini..."):
                boardroom_memo = generate_boardroom_pitch(basket, scenario_params, target_club=target_club)
                st.text_area("Boardroom Proposal", boardroom_memo, height=440)
        else:
            st.error(f"Optimization Infeasible: {status_msg}. The constraints are too restrictive for the available candidate pool. Try increasing the fee/wage budget or lowering the tactical production floors.")