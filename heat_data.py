"""
Loads the CDC Environmental Public Health Tracking extracts and Census poverty
estimates into tidy tables the dashboard can slice.

Heat events: county-level counts of extreme heat events (2+ consecutive days
with max temperature >= 90F, NOAA), summed here to state-year.
Health outcomes: state-year counts of heat-related hospitalizations and ER
visits (by age group and sex) and deaths (totals only).
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CDC_DIR = os.path.join(ROOT, "EDAV-Project", "data_clean")
POVERTY_PATH = os.path.join(ROOT, "data", "state_poverty.csv")

OUTCOMES = {
    "Hospitalizations": ("hosp_clean.csv", "Hosps"),
    "ER Visits": ("er_visits_clean.csv", "ER.Visits"),
    "Deaths": ("deaths_clean.csv", "Deaths"),
}
# Outcomes broken down by age and sex; deaths are only published as totals
DEMOGRAPHIC_OUTCOMES = ["Hospitalizations", "ER Visits"]
AGE_ORDER = ["0 TO 4", "5 TO 14", "15 TO 34", "35 TO 64", ">= 65"]
AGE_LABELS = {"0 TO 4": "0–4", "5 TO 14": "5–14", "15 TO 34": "15–34",
              "35 TO 64": "35–64", ">= 65": "65+"}


def load_heat_events() -> pd.DataFrame:
    """County-year extreme heat event counts."""
    return pd.read_csv(os.path.join(CDC_DIR, "heat_events_clean.csv"))


def load_outcomes() -> pd.DataFrame:
    """All three outcomes stacked into one long table:
    StateFIPS, State, Year, Outcome, Age_Group, Sex, Count."""
    frames = []
    for outcome, (filename, column) in OUTCOMES.items():
        df = pd.read_csv(os.path.join(CDC_DIR, filename))
        df = df.rename(columns={column: "Count"})
        for col in ("Age_Group", "Sex"):
            if col not in df:
                df[col] = "All"
        df["Outcome"] = outcome
        frames.append(df[["StateFIPS", "State", "Year", "Outcome", "Age_Group", "Sex", "Count"]])
    out = pd.concat(frames, ignore_index=True)
    out["Age_Group"] = out["Age_Group"].replace(AGE_LABELS)
    return out


def load_poverty() -> pd.DataFrame:
    """State-year poverty rate and population (Census SAIPE)."""
    return pd.read_csv(POVERTY_PATH)


def state_year_panel(outcomes: pd.DataFrame, heat: pd.DataFrame, poverty: pd.DataFrame,
                     outcome: str) -> pd.DataFrame:
    """One row per state-year with the outcome count, heat events, poverty rate,
    population and rate per 100k. Only state-years where the outcome is reported."""
    counts = (outcomes[outcomes["Outcome"] == outcome]
              .groupby(["StateFIPS", "State", "Year"], as_index=False)["Count"].sum())
    heat_sy = heat.groupby(["StateFIPS", "Year"], as_index=False)["EHE"].sum()
    panel = (counts.merge(heat_sy, on=["StateFIPS", "Year"], how="inner")
             .merge(poverty, on=["StateFIPS", "Year"], how="left"))
    panel["Rate_per_100k"] = panel["Count"] / panel["Population"] * 1e5
    return panel.sort_values(["State", "Year"]).reset_index(drop=True)
