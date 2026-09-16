import os
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint

def load_financial_market_data():
    path = os.path.join("data", "transfer_market_master.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Please run 'python 02_build_market_db.py' first."
        )
    return pd.read_csv(path)

def solve_transfer_portfolio(
    df,
    fee_budget_m=95.0,
    wage_ceiling_k_pw=180.0,
    target_signings_count=2,
    max_avg_age=24.0,
    min_prog_passes=10.0,
    min_defensive_actions=5.0,
    min_creation_threat=0.60
):
    """
    Solves a Mixed-Integer Linear Program (MILP) using SciPy to select the optimal
    basket of target_signings_count players maximizing composite tactical value 
    while respecting strict financial, age, and tactical floor constraints.
    """
    n = len(df)

    # 1. Objective Function: Composite Tactical Value (CTV)
    # Weights tactical metrics + youthful upside multiplier
    composite_value = (
        0.30 * df["progressive_passes_per90"] +
        0.25 * df["tackles_interceptions_per90"] +
        0.25 * (df["xag_per90"] + df["npxg_per90"]) * 5.0 +
        0.20 * (df["resale_potential_multiplier"] * 4.0)
    )
    # SciPy MILP minimizes by default; negate to maximize
    c = -composite_value.values

    # 2. Constraints Setup
    A_rows = []
    lhs_bounds = []
    rhs_bounds = []

    # Constraint 1: Transfer Fee Budget (Sum <= fee_budget_m)
    A_rows.append(df["market_value_eur_m"].values)
    lhs_bounds.append(0.0)
    rhs_bounds.append(fee_budget_m)

    # Constraint 2: Wage Ceiling (Sum <= wage_ceiling_k_pw)
    A_rows.append(df["wage_gbp_k_pw"].values)
    lhs_bounds.append(0.0)
    rhs_bounds.append(wage_ceiling_k_pw)

    # Constraint 3: Exact Number of Signings (Sum == target_signings_count)
    A_rows.append(np.ones(n))
    lhs_bounds.append(float(target_signings_count))
    rhs_bounds.append(float(target_signings_count))

    # Constraint 4: Tactical Floor - Progressive Passes (Sum >= min_prog_passes)
    A_rows.append(df["progressive_passes_per90"].values)
    lhs_bounds.append(min_prog_passes)
    rhs_bounds.append(np.inf)

    # Constraint 5: Tactical Floor - Defensive Actions (Sum >= min_defensive_actions)
    A_rows.append(df["tackles_interceptions_per90"].values)
    lhs_bounds.append(min_defensive_actions)
    rhs_bounds.append(np.inf)

    # Constraint 6: Tactical Floor - Combined xG + xAG (Sum >= min_creation_threat)
    combined_threat = (df["xag_per90"] + df["npxg_per90"]).values
    A_rows.append(combined_threat)
    lhs_bounds.append(min_creation_threat)
    rhs_bounds.append(np.inf)

    # Constraint 7: Age Cap - Mean Age <= max_avg_age => Sum(Age - max_avg_age) <= 0
    A_rows.append((df["age"] - max_avg_age).values)
    lhs_bounds.append(-np.inf)
    rhs_bounds.append(0.0)

    # Constraint 8: Positional Diversity - At most 1 player per specific position
    for pos in df["position"].unique():
        mask = (df["position"] == pos).astype(float).values
        A_rows.append(mask)
        lhs_bounds.append(0.0)
        rhs_bounds.append(1.0)

    # Assemble constraints matrix
    A = np.vstack(A_rows)
    constraints = LinearConstraint(A, lhs_bounds, rhs_bounds)

    # Variable type: all decision variables x_i must be binary integers {0, 1}
    integrality = np.ones(n)

    # Solve MILP
    res = milp(c=c, integrality=integrality, constraints=constraints)

    if not res.success:
        return None, res.status_message

    selected_indices = np.where(np.round(res.x) == 1)[0]
    portfolio = df.iloc[selected_indices].copy().reset_index(drop=True)
    return portfolio, "Optimal solution found"

if __name__ == "__main__":
    df = load_financial_market_data()
    print("--- Running Test Optimization Run ---")
    print("Scenario: Replacing a star midfielder with €90M budget and £170k/week wages")

    basket, message = solve_transfer_portfolio(
        df,
        fee_budget_m=90.0,
        wage_ceiling_k_pw=170.0,
        target_signings_count=2,
        max_avg_age=23.5,
        min_prog_passes=11.0,
        min_defensive_actions=7.0,
        min_creation_threat=0.40
    )

    if basket is not None:
        print(f"\n[STATUS] {message}")
        print(basket[["player_name", "position", "team", "age", "market_value_eur_m", "wage_gbp_k_pw", "progressive_passes_per90", "tackles_interceptions_per90"]])
        print(f"\nTotal Spent: €{basket['market_value_eur_m'].sum():.1f}M / €90.0M")
        print(f"Total Wages: £{basket['wage_gbp_k_pw'].sum():.1f}k / £170.0k per week")
        print(f"Average Age: {basket['age'].mean():.1f} years (Cap: ≤23.5)")
    else:
        print(f"[STATUS] Solver failed: {message}")