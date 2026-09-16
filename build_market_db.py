import os
import pandas as pd

def build_transfer_market_db():
    print("Enriching scouting database with financial & contract valuations...")
    
    # Financial data mapping: (estimated_fee_eur_m, wage_k_pw, contract_years, potential_multiplier)
    FINANCIAL_PROFILES = {
        # Goalkeepers
        "Alisson": (32.0, 210.0, 3, 0.95),
        "Ederson": (35.0, 180.0, 2, 0.95),
        "David Raya": (35.0, 100.0, 4, 1.05),
        "André Onana": (40.0, 120.0, 4, 1.05),
        "Gregor Kobel": (45.0, 85.0, 3, 1.15),
        "Guglielmo Vicario": (38.0, 75.0, 4, 1.10),
        "Diogo Costa": (50.0, 60.0, 3, 1.30),
        "Bart Verbruggen": (28.0, 40.0, 4, 1.40),

        # Center-Backs
        "Lisandro Martínez": (50.0, 120.0, 3, 1.10),
        "John Stones": (38.0, 250.0, 2, 0.90),
        "Alessandro Bastoni": (75.0, 130.0, 4, 1.20),
        "William Saliba": (85.0, 190.0, 4, 1.25),
        "Virgil van Dijk": (30.0, 220.0, 1, 0.85),
        "Jean-Clair Todibo": (40.0, 70.0, 4, 1.15),
        "Castello Lukeba": (45.0, 55.0, 4, 1.35),
        "Gonçalo Inácio": (45.0, 45.0, 3, 1.30),
        "Pau Cubarsí": (50.0, 35.0, 4, 1.45),
        "Jarrad Branthwaite": (65.0, 70.0, 3, 1.35),
        "Murillo": (40.0, 50.0, 4, 1.35),

        # Fullbacks & Wingbacks
        "Riccardo Calafiori": (45.0, 85.0, 4, 1.30),
        "Trent Alexander-Arnold": (70.0, 180.0, 1, 1.10),
        "Alphonso Davies": (55.0, 170.0, 1, 1.20),
        "Pedro Porro": (45.0, 90.0, 4, 1.15),
        "Jeremie Frimpong": (45.0, 80.0, 3, 1.25),
        "Federico Dimarco": (50.0, 95.0, 3, 1.05),
        "Destiny Udogie": (48.0, 75.0, 4, 1.35),

        # Defensive & Central Midfielders
        "Rodri": (110.0, 220.0, 3, 1.05),
        "Declan Rice": (105.0, 240.0, 4, 1.15),
        "Aurélien Tchouaméni": (90.0, 200.0, 4, 1.25),
        "João Palhinha": (50.0, 130.0, 3, 0.95),
        "Martín Zubimendi": (60.0, 95.0, 3, 1.15),
        "Alexis Mac Allister": (75.0, 150.0, 4, 1.15),
        "Bruno Guimarães": (85.0, 160.0, 4, 1.15),
        "Adam Wharton": (42.0, 55.0, 4, 1.40),
        "Kobbie Mainoo": (55.0, 45.0, 4, 1.45),
        "Eduardo Camavinga": (90.0, 140.0, 4, 1.35),

        # Attacking Midfielders
        "Florian Wirtz": (110.0, 160.0, 3, 1.40),
        "Martin Ødegaard": (100.0, 220.0, 4, 1.10),
        "Jude Bellingham": (140.0, 260.0, 5, 1.40),

        # Wingers
        "Bukayo Saka": (120.0, 230.0, 4, 1.25),
        "Vinícius Júnior": (150.0, 300.0, 4, 1.25),
        "Kylian Mbappé": (160.0, 400.0, 4, 1.15),
        "Michael Olise": (60.0, 110.0, 4, 1.30),
        "Nico Williams": (60.0, 120.0, 3, 1.30),
        "Bryan Mbeumo": (42.0, 75.0, 3, 1.15),
        "Antoine Semenyo": (32.0, 50.0, 3, 1.20),

        # Strikers
        "Erling Haaland": (180.0, 375.0, 3, 1.25),
        "Ollie Watkins": (65.0, 130.0, 4, 1.00),
        "Alexander Isak": (80.0, 140.0, 3, 1.20),
        "Rasmus Højlund": (60.0, 90.0, 4, 1.35),
        "Joshua Zirkzee": (45.0, 75.0, 4, 1.30),
        "Viktor Gyökeres": (70.0, 85.0, 4, 1.15)
    }

    base_path = os.path.join("data", "scouting_master_2024.csv")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Missing {base_path}. Please run 01_fetch_data.py first.")

    df = pd.read_csv(base_path)

    fee_list = []
    wage_list = []
    contract_list = []
    potential_list = []

    for _, row in df.iterrows():
        p_name = row["player_name"]
        profile = FINANCIAL_PROFILES.get(p_name, (30.0, 60.0, 3, 1.10))
        fee_list.append(profile[0])
        wage_list.append(profile[1])
        contract_list.append(profile[2])
        potential_list.append(profile[3])

    df["market_value_eur_m"] = fee_list
    df["wage_gbp_k_pw"] = wage_list
    df["contract_years_left"] = contract_list
    df["resale_potential_multiplier"] = potential_list

    out_path = os.path.join("data", "transfer_market_master.csv")
    df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Financial master created with {len(df)} players at: {out_path}")

if __name__ == "__main__":
    build_transfer_market_db()