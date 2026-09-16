import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BOARDROOM_PERSONA = """
You are the Technical Director & Head of Football Operations for an elite European football club.
You are delivering an executive capital-allocation and squad-planning proposal to the Club Chairman, Chief Executive, and Investment Board.

Tone: Highly strategic, fiscally disciplined, data-literate, and persuasive. Avoid generic football clichés; focus on operational risk mitigation, tactical efficiency, wage structure protection, and squad asset appreciation.

Format:
1. EXECUTIVE SUMMARY & CAPITAL EFFICIENCY (The macro financial case, cash preserved, wage bill relief)
2. TACTICAL WORKLOAD ABSORPTION (How the combined metrics compensate for the departure across progression and defensive recovery)
3. ASSET RESALE & AMORTIZATION THESIS (Age curves, contract runway, long-term squad value creation)
4. BOARDROOM RECOMMENDATION & NEXT ACTIONS (Concrete authorization requested)
"""

def generate_boardroom_pitch(portfolio_df, scenario_details, target_club="Manchester United"):
    api_key = os.getenv("GEMINI_API_KEY")
    total_spent = portfolio_df["market_value_eur_m"].sum()
    total_wages = portfolio_df["wage_gbp_k_pw"].sum()
    avg_age = portfolio_df["age"].mean()
    combined_prog = portfolio_df["progressive_passes_per90"].sum()
    combined_def = portfolio_df["tackles_interceptions_per90"].sum()

    financial_summary = {
        "budget_available_eur_m": scenario_details.get("fee_budget_m", 90.0),
        "total_spent_eur_m": round(total_spent, 1),
        "cash_reserve_preserved_eur_m": round(scenario_details.get("fee_budget_m", 90.0) - total_spent, 1),
        "wage_budget_k_pw": scenario_details.get("wage_ceiling_k_pw", 170.0),
        "committed_wages_k_pw": round(total_wages, 1),
        "wage_savings_k_pw": round(scenario_details.get("wage_ceiling_k_pw", 170.0) - total_wages, 1),
        "average_portfolio_age": round(avg_age, 1),
        "combined_prog_passes_per90": round(combined_prog, 1),
        "combined_defensive_actions_per90": round(combined_def, 1)
    }

    players_payload = portfolio_df[[
        "player_name", "position", "team", "league", "age",
        "market_value_eur_m", "wage_gbp_k_pw", "progressive_passes_per90",
        "tackles_interceptions_per90", "resale_potential_multiplier"
    ]].to_dict(orient="records")

    user_prompt = f"""
    TARGET CLUB: {target_club}
    RECRUITMENT SCENARIO: {scenario_details.get('scenario_title', 'Reinvestment of Outgoing Star Capital')}

    PORTFOLIO METRICS & FINANCIAL RECAP:
    {json.dumps(financial_summary, indent=2)}

    ACQUISITION TARGETS (SELECTED BY MIXED-INTEGER OPTIMIZER):
    {json.dumps(players_payload, indent=2)}
    """

    if api_key:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{BOARDROOM_PERSONA}\n\n{user_prompt}"
        )
        return response.text

    # High-level deterministic fallback if no API key is set
    return f"""================================================================================
EXECUTIVE BOARDROOM INVESTMENT PITCH | {target_club.upper()} FOOTBALL OPERATIONS
Scenario: Reinvestment Portfolio Optimization
================================================================================

1. EXECUTIVE SUMMARY & CAPITAL EFFICIENCY
- Total Capital Outlay: €{total_spent:.1f}M (Budget: €{scenario_details.get('fee_budget_m', 90.0):.1f}M | Preserved Cash: €{scenario_details.get('fee_budget_m', 90.0) - total_spent:.1f}M)
- Committed Weekly Wages: £{total_wages:.1f}k/pw (Savings: £{scenario_details.get('wage_ceiling_k_pw', 170.0) - total_wages:.1f}k/pw)
- Multi-asset diversification mitigates single-player catastrophic injury risk.

2. TACTICAL WORKLOAD ABSORPTION
- Combined Progressive Passes: {combined_prog:.1f}/90
- Combined Defensive Disruptions: {combined_def:.1f}/90
- Absorbs the operational volume of the outgoing asset while strengthening positional depth.

3. ASSET RESALE & AMORTIZATION THESIS
- Average Portfolio Age: {avg_age:.1f} years.
- Strong resale value runway on 4-year contract amortization schedules.

4. BOARDROOM RECOMMENDATION
- Approve immediate engagement with representatives of {[p['player_name'] for p in players_payload]}.
================================================================================"""

if __name__ == "__main__":
    from portfolio_optimizer import load_financial_market_data, solve_transfer_portfolio

    df = load_financial_market_data()
    scenario = {
        "scenario_title": "Star Midfielder Outgoing Re-investment",
        "fee_budget_m": 90.0,
        "wage_ceiling_k_pw": 170.0
    }
    basket, msg = solve_transfer_portfolio(df, fee_budget_m=90.0, wage_ceiling_k_pw=170.0)
    if basket is not None:
        print(generate_boardroom_pitch(basket, scenario, target_club="Manchester United"))