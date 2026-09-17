import pytest
import numpy as np
import pandas as pd

from similarity_engine import find_similar_players, load_data, POSITION_TO_COHORT
from portfolio_optimizer import load_financial_market_data, solve_transfer_portfolio


@pytest.fixture
def scouting_data():
    return load_data()


@pytest.fixture
def financial_data():
    return load_financial_market_data()


# -------------------------------------------------------------
# SIMILARITY ENGINE TESTS
# -------------------------------------------------------------
def test_similarity_score_bounds(scouting_data):
    """Test that similarity scores strictly reside in the [0, 100] percentile range."""
    results, _, _ = find_similar_players(scouting_data, reference_player_name="Rodri", top_n=5)
    
    assert not results.empty
    # Length should not exceed requested top_n, and must match available peers
    assert len(results) <= 5
    assert (results["similarity_score"] >= 0.0).all()
    assert (results["similarity_score"] <= 100.0).all()


def test_reference_player_self_exclusion(scouting_data):
    """Ensure the target player is never returned as their own similar candidate."""
    target_name = "Bukayo Saka"
    results, _, _ = find_similar_players(scouting_data, reference_player_name=target_name, top_n=5)
    
    assert target_name.lower() not in results["player_name"].str.lower().values


def test_age_and_league_filtering(scouting_data):
    """Verify that age limits and league filters are strictly respected."""
    max_age = 23
    selected_leagues = ["Premier League"]
    results, _, _ = find_similar_players(
        scouting_data,
        reference_player_name="Erling Haaland",
        max_age=max_age,
        selected_leagues=selected_leagues,
        top_n=5
    )
    
    if not results.empty:
        assert (results["age"] <= max_age).all()
        assert (results["league"] == "Premier League").all()


# -------------------------------------------------------------
# PORTFOLIO OPTIMIZER (MILP) TESTS
# -------------------------------------------------------------
def test_milp_solver_budget_constraints(financial_data):
    """Ensure the MILP solution never exceeds the allocated capital fee or wage budget."""
    budget_fee = 90.0
    wage_ceiling = 170.0
    signings = 2

    basket, status = solve_transfer_portfolio(
        financial_data,
        fee_budget_m=budget_fee,
        wage_ceiling_k_pw=wage_ceiling,
        target_signings_count=signings,
        max_avg_age=24.0,
        min_prog_passes=8.0,
        min_defensive_actions=5.0
    )

    assert basket is not None, f"Solver failed: {status}"
    assert len(basket) == signings
    assert basket["market_value_eur_m"].sum() <= budget_fee
    assert basket["wage_gbp_k_pw"].sum() <= wage_ceiling


def test_milp_positional_diversity(financial_data):
    """Ensure the optimizer does not purchase two players in the exact same position."""
    basket, _ = solve_transfer_portfolio(
        financial_data,
        fee_budget_m=100.0,
        wage_ceiling_k_pw=200.0,
        target_signings_count=2
    )

    if basket is not None:
        unique_positions = basket["position"].nunique()
        assert unique_positions == len(basket), "Duplicate positions found in portfolio!"