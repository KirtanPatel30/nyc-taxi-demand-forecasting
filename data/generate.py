"""
data/generate.py
Generates a realistic NYC TLC-style taxi trip dataset at scale.
Mirrors real TLC data columns: pickup/dropoff datetime, location IDs,
passenger count, trip distance, fare amount.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# 63 NYC taxi zones (simplified)
ZONES = list(range(1, 64))
ZONE_NAMES = {
    1: "Newark Airport", 2: "Jamaica Bay", 3: "Allerton/Pelham Gardens",
    4: "Alphabet City", 5: "Arden Heights", 6: "Arrochar/Fort Wadsworth",
    7: "Astoria", 8: "Astoria Park", 9: "Auburndale", 10: "Baisley Park",
    11: "Bath Beach", 12: "Battery Park", 13: "Bay Ridge", 14: "Bay Terrace",
    15: "Bayside", 16: "Bedford", 17: "Bedford Park", 18: "Bellerose",
    19: "Belmont", 20: "Bensonhurst East", 21: "Bensonhurst West",
    22: "Bloomfield/Emerson Hill", 23: "Bloomingdale", 24: "Boerum Hill",
    25: "Borough Park", 26: "Breezy Point/Fort Tilden", 27: "Briarwood/Jamaica Hills",
    28: "Brighton Beach", 29: "Broad Channel", 30: "Bronx Park",
    31: "Bronxdale", 32: "Brooklyn Heights", 33: "Brooklyn Navy Yard",
    34: "Brownsville", 35: "Bushwick North", 36: "Bushwick South",
    37: "Cambria Heights", 38: "Canarsie", 39: "Carroll Gardens",
    40: "Central Harlem", 41: "Central Harlem North", 42: "Central Park",
    43: "Charleston/Tottenville", 44: "Chinatown", 45: "City Island",
    46: "Claremont/Bathgate", 47: "Clinton East", 48: "Clinton Hill",
    49: "Clinton West", 50: "Co-Op City", 51: "College Point",
    52: "Columbia Street", 53: "Coney Island", 54: "Corona",
    55: "Corona", 56: "Country Club", 57: "Crotona Park",
    58: "Crotona Park East", 59: "Crown Heights North", 60: "Crown Heights South",
    61: "Cypress Hills", 62: "Douglas Manor", 63: "Douglaston",
}

# High-demand zones (Midtown, airports, downtown)
HIGH_DEMAND_ZONES = [4, 12, 42, 47, 49, 1, 7, 32, 40, 44]

# Hour-of-day demand multipliers (rush hours higher)
HOURLY_DEMAND = {
    0: 0.3, 1: 0.2, 2: 0.15, 3: 0.1, 4: 0.1, 5: 0.15,
    6: 0.4, 7: 0.8, 8: 1.0, 9: 0.9, 10: 0.7, 11: 0.7,
    12: 0.8, 13: 0.8, 14: 0.7, 15: 0.8, 16: 0.9, 17: 1.0,
    18: 1.0, 19: 0.9, 20: 0.8, 21: 0.8, 22: 0.7, 23: 0.5,
}


def generate_trips(n=1_000_000, days=90):
    print(f"Generating {n:,} NYC taxi trips over {days} days...")

    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(minutes=int(m))
                  for m in np.random.randint(0, days * 24 * 60, n)]
    timestamps.sort()

    hours = np.array([t.hour for t in timestamps])
    hour_multiplier = np.array([HOURLY_DEMAND[h] for h in hours])

    # Zone assignment — high demand zones get more pickups
    zone_probs = np.ones(len(ZONES))
    for z in HIGH_DEMAND_ZONES:
        zone_probs[ZONES.index(z)] = 5.0
    zone_probs /= zone_probs.sum()

    pickup_zones  = np.random.choice(ZONES, n, p=zone_probs)
    dropoff_zones = np.random.choice(ZONES, n)

    # Trip distance correlated with zone distance (simplified)
    distances = np.abs(pickup_zones - dropoff_zones) * 0.15 + np.random.exponential(1.5, n)
    distances = distances.clip(0.1, 30.0).round(2)

    # Fare amount correlated with distance
    fares = (2.5 + distances * 2.5 + np.random.normal(0, 1.5, n)).clip(3.0, 150.0).round(2)

    # Passenger count
    passengers = np.random.choice([1, 2, 3, 4, 5, 6], n, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02])

    # Trip duration in minutes
    durations = (distances * 3 + np.random.normal(5, 3, n)).clip(1, 120).astype(int)
    dropoff_timestamps = [
        timestamps[i] + timedelta(minutes=int(durations[i]))
        for i in range(n)
    ]

    df = pd.DataFrame({
        "pickup_datetime":  [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "dropoff_datetime": [t.strftime("%Y-%m-%d %H:%M:%S") for t in dropoff_timestamps],
        "pickup_zone":      pickup_zones,
        "dropoff_zone":     dropoff_zones,
        "passenger_count":  passengers,
        "trip_distance":    distances,
        "fare_amount":      fares,
        "trip_duration_min": durations,
    })

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['pickup_datetime'].min()} → {df['pickup_datetime'].max()}")
    return df


def save_data(df):
    out = RAW_DIR / "trips.csv"
    df.to_csv(out, index=False)
    print(f"Saved to {out}")
    return out


if __name__ == "__main__":
    df = generate_trips()
    save_data(df)
    print(df.head())
