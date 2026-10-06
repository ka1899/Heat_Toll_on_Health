"""
Heat-health model and climate scenario simulator.

Model (fit separately for each outcome):

    log E[count] = log(population)                       # offset -> models a rate
                 + b1 * log(1 + heat_events)             # heat dose-response
                 + b2 * poverty_c                        # poverty (centered)
                 + b3 * log(1 + heat_events) * poverty_c # does poverty amplify heat?
                 + state fixed effects + year fixed effects

Fit as a quasi-Poisson GLM (variance scaled for overdispersion) with standard
errors clustered by state. State effects absorb fixed differences between
states (climate, reporting systems); year effects absorb nationwide shocks
(coding changes, COVID). So b1 and b3 are identified from within-state,
year-to-year variation in heat.

Because heat enters as log(1 + events), raising heat events by x% multiplies
the expected count by roughly (1 + x)^(b1 + b3 * poverty_c): always increasing
when the slope is positive, with diminishing returns, and steeper in
higher-poverty states when b3 > 0.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

FORMULA = "Count ~ log_heat * poverty_c + C(State) + C(Year)"
TERMS = ["log_heat", "poverty_c", "log_heat:poverty_c"]
Z95 = 1.959964


@dataclass
class HeatHealthModel:
    outcome: str
    result: object          # statsmodels GLMResults
    poverty_mean: float
    n_obs: int
    n_states: int

    def coef_table(self) -> pd.DataFrame:
        ci = self.result.conf_int().loc[TERMS]
        return pd.DataFrame({
            "Term": ["Heat (log events)", "Poverty rate (pts)", "Heat × poverty"],
            "Estimate": self.result.params[TERMS].values,
            "CI low": ci[0].values,
            "CI high": ci[1].values,
            "p-value": self.result.pvalues[TERMS].values,
        })

    def heat_slope(self, poverty_rate):
        """Heat elasticity (and its std. error) at a given poverty rate."""
        p = self.result.params
        v = self.result.cov_params()
        pc = np.asarray(poverty_rate, dtype=float) - self.poverty_mean
        slope = p["log_heat"] + p["log_heat:poverty_c"] * pc
        var = (v.loc["log_heat", "log_heat"]
               + pc ** 2 * v.loc["log_heat:poverty_c", "log_heat:poverty_c"]
               + 2 * pc * v.loc["log_heat", "log_heat:poverty_c"])
        return slope, np.sqrt(var)


def prepare(panel: pd.DataFrame, poverty_mean: float) -> pd.DataFrame:
    df = panel.dropna(subset=["Count", "EHE", "Poverty_Rate", "Population"]).copy()
    df["log_heat"] = np.log1p(df["EHE"])
    df["poverty_c"] = df["Poverty_Rate"] - poverty_mean
    return df


def fit_model(panel: pd.DataFrame, outcome: str) -> HeatHealthModel:
    poverty_mean = panel["Poverty_Rate"].mean()
    df = prepare(panel, poverty_mean)
    groups = df["State"].astype("category").cat.codes
    result = smf.glm(FORMULA, data=df, family=sm.families.Poisson(),
                     offset=np.log(df["Population"])).fit(
        scale="X2", cov_type="cluster", cov_kwds={"groups": groups})
    return HeatHealthModel(outcome, result, poverty_mean, len(df), df["State"].nunique())


def simulate_heat_increase(panel: pd.DataFrame, model: HeatHealthModel,
                           pct_increase: float, recent_years: int = 5) -> pd.DataFrame:
    """Projected change in a typical recent year if heat events rise by pct_increase %.

    For each state, the baseline is its average heat events and outcome count
    over the last `recent_years` reported years, and its latest poverty rate.
    """
    last = panel["Year"].max()
    recent = panel[panel["Year"] > last - recent_years]
    base = recent.groupby("State").agg(
        Heat_Events=("EHE", "mean"),
        Baseline=("Count", "mean"),
        Poverty_Rate=("Poverty_Rate", "last"),
        Population=("Population", "last"),
    ).reset_index()

    factor = 1 + pct_increase / 100
    dose = np.log1p(base["Heat_Events"] * factor) - np.log1p(base["Heat_Events"])
    slope, se = model.heat_slope(base["Poverty_Rate"])
    log_rr = slope * dose
    base["Rate_Ratio"] = np.exp(log_rr)
    base["Projected"] = base["Baseline"] * base["Rate_Ratio"]
    base["Extra"] = base["Projected"] - base["Baseline"]
    base["Extra_low"] = base["Baseline"] * (np.exp(log_rr - Z95 * se * dose) - 1)
    base["Extra_high"] = base["Baseline"] * (np.exp(log_rr + Z95 * se * dose) - 1)
    base["Pct_Change"] = (base["Rate_Ratio"] - 1) * 100
    return base.sort_values("Extra", ascending=False).reset_index(drop=True)


def sensitivity_curve(model: HeatHealthModel, poverty_range, pct_increase: float) -> pd.DataFrame:
    """% change in the outcome from a pct_increase rise in heat, across poverty
    rates (evaluated where heat events are large, so dose ~= log(1 + x))."""
    pov = np.linspace(*poverty_range, 50)
    slope, se = model.heat_slope(pov)
    dose = np.log(1 + pct_increase / 100)
    return pd.DataFrame({
        "Poverty_Rate": pov,
        "Pct_Change": (np.exp(slope * dose) - 1) * 100,
        "Low": (np.exp((slope - Z95 * se) * dose) - 1) * 100,
        "High": (np.exp((slope + Z95 * se) * dose) - 1) * 100,
    })


if __name__ == "__main__":
    from heat_data import load_heat_events, load_outcomes, load_poverty, state_year_panel

    outcomes, heat, poverty = load_outcomes(), load_heat_events(), load_poverty()
    for outcome in ["Hospitalizations", "ER Visits", "Deaths"]:
        panel = state_year_panel(outcomes, heat, poverty, outcome)
        m = fit_model(panel, outcome)
        print(f"\n=== {outcome}: {m.n_obs} state-years, {m.n_states} states ===")
        print(m.coef_table().round(4).to_string(index=False))
        sim = simulate_heat_increase(panel, m, 50)
        print(f"+50% heat events -> {sim['Extra'].sum():,.0f} extra per year across these states")
