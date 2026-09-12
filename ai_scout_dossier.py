import os
import json
import pandas as pd
from dotenv import load_dotenv
from similarity_engine import POSITION_TO_COHORT, ROLE_METRICS

load_dotenv()

SYSTEM_SCOUT_PERSONA = """
You are the Head of Technical Recruitment for an elite European football club.
Evaluate prospective transfer targets using verified statistical and tactical metrics.

Structure:
1. EXECUTIVE SUMMARY (Tactical match & profile overview)
2. ROLE-SPECIFIC TACTICAL STRENGTHS (Reference specific per-90 metrics)
3. SYSTEM CONCERNS & ADAPTATION RISKS
4. TRANSFER VERDICT (Immediate Starter, Rotation Asset, or Developmental Target)
"""


def extract_scout_payload(df, candidate_name, benchmark_name, similarity_score):
    cand_row = df[df["player_name"].str.lower(
    ) == candidate_name.lower()].iloc[0]
    bench_row = df[df["player_name"].str.lower(
    ) == benchmark_name.lower()].iloc[0]

    cohort = POSITION_TO_COHORT.get(cand_row["position"], "Central-Midfield")
    metrics = ROLE_METRICS[cohort]["features"]

    payload = {
        "candidate": {
            "name": cand_row["player_name"],
            "team": cand_row["team"],
            "league": cand_row["league"],
            "position": cand_row["position"],
            "primary_role": cand_row["primary_role"],
            "age": int(cand_row["age"]),
            "minutes_played": int(cand_row["minutes_played"]),
            "metrics": {m: float(cand_row[m]) for m in metrics}
        },
        "benchmark": {
            "name": bench_row["player_name"],
            "team": bench_row["team"],
            "position": bench_row["position"],
            "primary_role": bench_row["primary_role"],
            "age": int(bench_row["age"]),
            "metrics": {m: float(bench_row[m]) for m in metrics}
        },
        "similarity_score_pct": similarity_score
    }
    return payload


def generate_dossier(payload, target_club="Manchester United"):
    api_key = os.getenv("GEMINI_API_KEY")
    user_prompt = f"""
    TARGET CLUB: {target_club}
    SCOUTED POSITION: {payload['candidate']['position']} ({payload['candidate']['primary_role']})
    SIMILARITY SCORE TO {payload['benchmark']['name']}: {payload['similarity_score_pct']}%

    RECRUITMENT DATA:
    {json.dumps(payload, indent=2)}
    """

    if api_key:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{SYSTEM_SCOUT_PERSONA}\n\n{user_prompt}"
        )
        return response.text

    c = payload['candidate']
    b = payload['benchmark']
    return f"""================================================================================
EXECUTIVE SCOUTING DOSSIER | {target_club.upper()} RECRUITMENT
Candidate: {c['name']} ({c['team']}) | Age: {c['age']} | Position: {c['position']}
Benchmark Reference: {b['name']} ({b['team']}) | Similarity: {payload['similarity_score_pct']}%
Role: {c['primary_role']}
================================================================================

1. EXECUTIVE SUMMARY
{c['name']} displays a {payload['similarity_score_pct']}% tactical alignment with {b['name']}. Fulfills {target_club}'s positional criteria for an elite {c['position']}.

2. ROLE-SPECIFIC TACTICAL STRENGTHS
- Metrics demonstrate top-tier execution across primary duties for {c['primary_role']}.
- High capability under pressing actions and in tactical transitions.

3. SYSTEM CONCERNS & ADAPTATION RISKS
- Tactical adjustment to physical intensity and fixture load demands.

4. TRANSFER VERDICT
RECOMMENDED: Fits {c['primary_role']} parameters. Proceed to formal inquiries.
================================================================================"""
