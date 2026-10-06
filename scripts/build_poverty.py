"""
Build data/state_poverty.csv from the Census Bureau's SAIPE program
(Small Area Income & Poverty Estimates), 2000-2022.

The SAIPE API now requires a key, so this reads the public bulk text files
instead. Raw files are cached in data_external/ (git-ignored) and downloaded
on first run.

    python scripts/build_poverty.py
"""
import os
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data_external")
OUT_PATH = os.path.join(ROOT, "data", "state_poverty.csv")
BASE_URL = "https://www2.census.gov/programs-surveys/saipe/datasets"


def raw_url(year: int) -> str:
    yy = f"{year % 100:02d}"
    # 2000-2003 were published as .dat, later years as .txt (same layout)
    name = f"est{yy}all.dat" if year <= 2003 else f"est{yy}all.txt"
    return f"{BASE_URL}/{year}/{year}-state-and-county/{name}"


def fetch(year: int) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"est{year % 100:02d}.txt")
    if not os.path.exists(path):
        print(f"Downloading SAIPE {year}...")
        urllib.request.urlretrieve(raw_url(year), path)
    return path


def parse_states(path: str, year: int) -> pd.DataFrame:
    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            tokens = line.split()
            state_fips, county_fips = int(tokens[0]), int(tokens[1])
            # State totals have county code 0; FIPS 0 is the US total
            if county_fips != 0 or state_fips == 0:
                continue
            poor_count = float(tokens[2])
            poverty_rate = float(tokens[5])
            rows.append({
                "StateFIPS": state_fips,
                "Year": year,
                "Poverty_Rate": poverty_rate,
                # SAIPE gives the count in poverty and the rate, which together
                # imply the size of the population for whom poverty is measured
                "Population": round(poor_count / (poverty_rate / 100)),
            })
    return pd.DataFrame(rows)


def main():
    frames = [parse_states(fetch(y), y) for y in range(2000, 2023)]
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
