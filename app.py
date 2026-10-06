import os

import altair as alt
import groq
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from heat_data import (AGE_LABELS, AGE_ORDER, DEMOGRAPHIC_OUTCOMES, OUTCOMES,
                       load_heat_events, load_outcomes, load_poverty, state_year_panel)
from ml_model import fit_model, sensitivity_curve, simulate_heat_increase

load_dotenv()
st.set_page_config(page_title="Heat's Toll on Health", page_icon="🔥", layout="wide")

US_STATES_TOPO = "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/us-10m.json"
HEAT_COLOR = "#e03131"
OUTCOME_COLORS = {"Hospitalizations": "#1971c2", "ER Visits": "#0c8599", "Deaths": "#7048e8"}
SEX_COLORS = alt.Scale(domain=["Female", "Male"], range=["#e8590c", "#1c7ed6"])
AGES = [AGE_LABELS[a] for a in AGE_ORDER]


def compact(n: float) -> str:
    """106509 -> '106.5K' so headline numbers fit their cards."""
    for div, suffix in ((1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.1f}{suffix}"
    return f"{n:,.0f}"


# ---------------------------------------------------------------- data + model
@st.cache_data
def load_all():
    return load_outcomes(), load_heat_events(), load_poverty()


@st.cache_resource
def get_model(outcome: str):
    outcomes, heat, poverty = load_all()
    return fit_model(state_year_panel(outcomes, heat, poverty, outcome), outcome)


outcomes_df, heat_df, poverty_df = load_all()
ALL_STATES = sorted(heat_df["State"].unique())
YEAR_MIN, YEAR_MAX = int(heat_df["Year"].min()), int(heat_df["Year"].max())


# ---------------------------------------------------------------- sidebar slicers
def reset_filters():
    for key in ("states", "years", "ages", "sexes"):
        st.session_state.pop(key, None)


with st.sidebar:
    st.header("Filters")
    outcome = st.segmented_control("Health outcome", list(OUTCOMES), default="Hospitalizations",
                                   key="outcome") or "Hospitalizations"
    states = st.multiselect("States", ALL_STATES, key="states",
                            placeholder=f"All {len(ALL_STATES)} states")
    years = st.slider("Years", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX), key="years")
    has_demographics = outcome in DEMOGRAPHIC_OUTCOMES
    ages = st.pills("Age group", AGES, selection_mode="multi", default=AGES, key="ages",
                    disabled=not has_demographics)
    sexes = st.pills("Sex", ["Female", "Male"], selection_mode="multi",
                     default=["Female", "Male"], key="sexes", disabled=not has_demographics)
    if not has_demographics:
        st.caption("Deaths are only published as totals, so age and sex filters don't apply.")
    st.button("Reset filters", on_click=reset_filters, width="stretch", key="reset")
    st.divider()
    st.caption("Sources: CDC Environmental Public Health Tracking (extreme heat days, heat-related health "
               "outcomes) · U.S. Census Bureau SAIPE (poverty, population).")

selected_states = states or ALL_STATES
mask = (
    (outcomes_df["Outcome"] == outcome)
    & outcomes_df["State"].isin(selected_states)
    & outcomes_df["Year"].between(*years)
)
if has_demographics:
    mask &= outcomes_df["Age_Group"].isin(ages or AGES) & outcomes_df["Sex"].isin(sexes or ["Female", "Male"])
filtered = outcomes_df[mask]
heat_filtered = heat_df[heat_df["State"].isin(selected_states) & heat_df["Year"].between(*years)]
panel = state_year_panel(filtered, heat_filtered, poverty_df, outcome)

# ---------------------------------------------------------------- header
st.title("🔥 Heat's Toll on Health")
scope = "all states" if not states else (", ".join(states) if len(states) <= 3 else f"{len(states)} states")
label = f"heat-related {outcome.lower()}"
st.markdown(f"Extreme heat and **{label}** · {scope} · {years[0]}–{years[1]}")

if panel.empty:
    st.warning(f"No {outcome.lower()} are reported for this selection. Try other states or years.")
    st.stop()

tab_overview, tab_sim, tab_ai, tab_nyc, tab_methods = st.tabs(
    ["📊 Overview", "🌡️ Heat Scenario Simulator", "🤖 AI Analyst", "🗽 NYC Spotlight", "📖 Methods"])

color = OUTCOME_COLORS[outcome]

# ================================================================ OVERVIEW
with tab_overview:
    yearly = panel.groupby("Year").agg(Count=("Count", "sum"), Population=("Population", "sum"),
                                       States=("State", "nunique")).reset_index()
    # Heat for the selection = average heat days across all counties in the reporting states
    county_heat = heat_filtered[heat_filtered["State"].isin(panel["State"].unique())]
    yearly = yearly.merge(county_heat.groupby("Year", as_index=False)["EHE"].mean()
                          .rename(columns={"EHE": "Heat_Days"}), on="Year", how="left")
    yearly["Rate"] = yearly["Count"] / yearly["Population"] * 1e5

    n_years = min(3, len(yearly))
    first, last = yearly.head(n_years), yearly.tail(n_years)
    rate_change = (last["Rate"].mean() / first["Rate"].mean() - 1) * 100 if len(yearly) > n_years else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total cases", compact(yearly["Count"].sum()),
              chart_data=yearly["Count"].tolist(), chart_type="bar", border=True,
              help=f"{yearly['Count'].sum():,.0f} {label} across the selected states, years, ages and sexes.")
    k2.metric("Rate per 100k", f"{yearly['Rate'].mean():.1f}",
              delta=f"{rate_change:+.0f}%" if rate_change is not None else None,
              delta_color="inverse", chart_data=yearly["Rate"].round(2).tolist(), border=True,
              help=f"Average yearly {label} per 100k residents of the reporting states. The % change compares "
                   f"the last {n_years} years with the first {n_years} years in the range.")
    k3.metric("Heat days", f"{yearly['Heat_Days'].mean():.0f} / yr",
              chart_data=yearly["Heat_Days"].round(1).tolist(), border=True,
              help="Extreme heat days per county per year, on average (CDC/NOAA). "
                   "Averaged across counties, not summed: one heat wave hitting 500 counties is still one heat wave.")
    k4.metric("States", f"{panel['State'].nunique()}", border=True,
              help="States reporting this outcome. Not every state reports every outcome every year, so totals partly reflect coverage. "
                   "The per-100k rate adjusts for this.")

    # --- trend: heat days above, outcome rate below, shared x-axis
    x = alt.X("Year:O", title=None, axis=alt.Axis(labelAngle=0, values=list(range(2000, 2023, 2))))
    heat_trend = alt.Chart(yearly).mark_area(color=HEAT_COLOR, opacity=0.25, line={"color": HEAT_COLOR}).encode(
        x=x, y=alt.Y("Heat_Days:Q", title="Days"),
        tooltip=["Year", alt.Tooltip("Heat_Days:Q", title="Heat days per county", format=".1f")]
    ).properties(height=130, title="Extreme heat days per county (average)")
    rate_trend = alt.Chart(yearly).mark_bar(color=color).encode(
        x=x, y=alt.Y("Rate:Q", title="Per 100k"),
        tooltip=["Year", alt.Tooltip("Count:Q", title=outcome, format=","),
                 alt.Tooltip("Rate:Q", title="Per 100k", format=".2f"),
                 alt.Tooltip("States:Q", title="States reporting")]
    ).properties(height=200, title=f"Heat-related {outcome.lower()} per 100k residents")
    st.altair_chart(alt.vconcat(heat_trend, rate_trend, spacing=8).resolve_scale(x="shared"), width="stretch")

    # --- map + ranking
    by_state = panel.groupby(["StateFIPS", "State"]).agg(
        Count=("Count", "sum"), Heat_Days=("Heat_Days", "mean"), Population=("Population", "mean"),
        Poverty_Rate=("Poverty_Rate", "mean"), Years=("Year", "nunique")).reset_index()
    by_state["Rate"] = by_state["Count"] / by_state["Years"] / by_state["Population"] * 1e5

    map_metrics = {
        f"{outcome} per 100k (yearly avg)": ("Rate", "orangered", ".2f"),
        "Heat days per county (yearly avg)": ("Heat_Days", "reds", ".0f"),
        "Poverty rate (%)": ("Poverty_Rate", "purples", ".1f"),
    }
    col_map, col_rank = st.columns([3, 2])
    with col_map:
        metric_label = st.segmented_control("Map shows", list(map_metrics), default=list(map_metrics)[0],
                                            key="map_metric") or list(map_metrics)[0]
        field, scheme, fmt = map_metrics[metric_label]
        geo = alt.topo_feature(US_STATES_TOPO, "states")
        lookup_cols = ["State", "Rate", "Heat_Days", "Poverty_Rate", "Count", "Years"]
        background = alt.Chart(geo).mark_geoshape(fill="#adb5bd", opacity=0.25, stroke="white", strokeWidth=0.5)
        states_layer = alt.Chart(geo).mark_geoshape(stroke="white", strokeWidth=0.7).transform_lookup(
            lookup="id", from_=alt.LookupData(by_state, "StateFIPS", lookup_cols)
        ).transform_filter("isValid(datum.State)").encode(
            color=alt.Color(f"{field}:Q", scale=alt.Scale(scheme=scheme), title=None,
                            legend=alt.Legend(orient="bottom", gradientLength=260, format=fmt)),
            tooltip=[alt.Tooltip("State:N"),
                     alt.Tooltip("Rate:Q", title=f"{outcome} per 100k/yr", format=".2f"),
                     alt.Tooltip("Heat_Days:Q", title="Heat days/county/yr", format=".0f"),
                     alt.Tooltip("Poverty_Rate:Q", title="Poverty rate %", format=".1f"),
                     alt.Tooltip("Years:Q", title="Years reported")])
        st.altair_chart((background + states_layer).project("albersUsa").properties(height=380),
                        width="stretch")
        st.caption("Grey states are not in the CDC tracking extract for this outcome.")

    with col_rank:
        st.markdown(f"**Highest {label} rates**")
        top = by_state.nlargest(12, "Rate")
        st.altair_chart(alt.Chart(top).mark_bar(color=color).encode(
            x=alt.X("Rate:Q", title="Per 100k residents per year"),
            y=alt.Y("State:N", sort="-x", title=None),
            tooltip=["State", alt.Tooltip("Rate:Q", format=".2f"), alt.Tooltip("Count:Q", format=",")]
        ).properties(height=380), width="stretch")

    # --- state x year heatmap
    st.markdown(f"**Heat-related {outcome.lower()} per 100k by state and year**")
    order = by_state.sort_values("Rate", ascending=False)["State"].tolist()
    st.altair_chart(alt.Chart(panel).mark_rect().encode(
        x=alt.X("Year:O", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("State:N", sort=order, title=None),
        # Cap colors at the 95th percentile so one extreme state doesn't wash out the rest
        color=alt.Color("Rate_per_100k:Q", title="Per 100k", scale=alt.Scale(
            scheme="orangered", domain=[0, float(panel["Rate_per_100k"].quantile(0.95))], clamp=True)),
        tooltip=["State", "Year", alt.Tooltip("Count:Q", title=outcome, format=","),
                 alt.Tooltip("Rate_per_100k:Q", title="Per 100k", format=".2f"),
                 alt.Tooltip("Heat_Days:Q", title="Heat days/county", format=".1f")]
    ).properties(height=max(160, 16 * len(order))), width="stretch")
    st.caption("Blank cells: the state did not report that year (or deaths were suppressed because there were fewer than 10).")

    col_demo, col_scatter = st.columns(2)
    with col_demo:
        st.markdown("**Who is affected**")
        if has_demographics:
            demo = filtered.groupby(["Age_Group", "Sex"], as_index=False)["Count"].sum()
            st.altair_chart(alt.Chart(demo).mark_bar().encode(
                x=alt.X("Age_Group:N", sort=AGES, title="Age group", axis=alt.Axis(labelAngle=0)),
                xOffset="Sex:N",
                y=alt.Y("Count:Q", title=outcome),
                color=alt.Color("Sex:N", scale=SEX_COLORS, legend=alt.Legend(orient="top", title=None)),
                tooltip=["Age_Group", "Sex", alt.Tooltip("Count:Q", format=",")]
            ).properties(height=300), width="stretch")
        else:
            st.info("Age and sex breakdowns are available for hospitalizations and ER visits only.")

    with col_scatter:
        st.markdown("**Hotter years, more cases?**")
        pts = panel[(panel["Heat_Days"] > 0) & (panel["Count"] > 0)]
        st.altair_chart(alt.Chart(pts).mark_circle(size=55, opacity=0.75).encode(
            x=alt.X("Heat_Days:Q", scale=alt.Scale(type="log"), title="Heat days per county that year (log)"),
            y=alt.Y("Rate_per_100k:Q", scale=alt.Scale(type="log"), title=f"{outcome} per 100k (log)"),
            color=alt.Color("Poverty_Rate:Q", scale=alt.Scale(scheme="purples"), title="Poverty %",
                            legend=alt.Legend(orient="top")),
            tooltip=["State", "Year", alt.Tooltip("Heat_Days:Q", title="Heat days/county", format=".1f"),
                     alt.Tooltip("Rate_per_100k:Q", format=".2f"), alt.Tooltip("Poverty_Rate:Q", format=".1f")]
        ).properties(height=300), width="stretch")
        st.caption("Each dot is one state-year. Both axes are logarithmic.")

    with st.expander("View and download the filtered data"):
        table = panel.rename(columns={"Count": outcome, "Heat_Days": "Heat_Days_per_County"}).drop(columns="StateFIPS")
        st.dataframe(table, hide_index=True, column_config={
            "Year": st.column_config.NumberColumn(format="%d"),
            "Rate_per_100k": st.column_config.NumberColumn("Per 100k", format="%.2f"),
            "Population": st.column_config.NumberColumn(format="localized"),
        })
        st.download_button("Download CSV", table.to_csv(index=False), "heat_health_filtered.csv", "text/csv")

# ================================================================ SIMULATOR
with tab_sim:
    model = get_model(outcome)
    st.markdown(
        f"What if extreme heat days became more common? This projects **{label}** in a typical "
        "recent year (each state's last 5 reported years) using a statistical model fit on "
        f"{model.n_obs} state-years from {model.n_states} states. Use the sidebar to choose which states to show.")

    pct = st.slider("Increase in extreme heat days", 0, 200, 50, step=10, format="+%d%%", key="pct",
                    help="Extreme heat days in every county scale up by this percentage.")
    full_panel = state_year_panel(outcomes_df[outcomes_df["Outcome"] == outcome], heat_df, poverty_df, outcome)
    sim = simulate_heat_increase(full_panel, model, pct)
    sim = sim[sim["State"].isin(selected_states)]

    if pct == 0:
        st.info("Move the slider to add heat days.")
    else:
        extra, baseline = sim["Extra"].sum(), sim["Baseline"].sum()
        s1, s2, s3 = st.columns(3)
        s1.metric(f"Extra {outcome.lower()} per year", f"+{extra:,.0f}", border=True,
                  help="Approximate 95% range: "
                       f"{sim['Extra_low'].sum():,.0f} to {sim['Extra_high'].sum():,.0f}")
        s2.metric("Change vs. today", f"{extra / baseline * 100:+.1f}%", border=True,
                  help=f"Baseline: {baseline:,.0f} per year across the selected states.")
        top_state = sim.iloc[0]
        s3.metric("Largest increase", top_state["State"], f"+{top_state['Extra']:,.0f} per year",
                  delta_color="inverse", border=True)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**Projected extra {outcome.lower()} per year, by state**")
            bars = alt.Chart(sim).encode(y=alt.Y("State:N", sort="-x", title=None))
            st.altair_chart((bars.mark_bar().encode(
                x=alt.X("Extra:Q", title=f"Extra {outcome.lower()} per year"),
                color=alt.Color("Poverty_Rate:Q", scale=alt.Scale(scheme="purples"), title="Poverty %",
                                legend=alt.Legend(orient="top")),
                tooltip=["State", alt.Tooltip("Baseline:Q", format=",.0f"),
                         alt.Tooltip("Projected:Q", format=",.0f"),
                         alt.Tooltip("Pct_Change:Q", title="% change", format="+.1f"),
                         alt.Tooltip("Extra_low:Q", title="95% low", format=",.0f"),
                         alt.Tooltip("Extra_high:Q", title="95% high", format=",.0f")])
                + bars.mark_rule(color="#868e96").encode(x="Extra_low:Q", x2="Extra_high:Q")
            ).properties(height=max(200, 18 * len(sim))), width="stretch")
            st.caption("Grey lines are 95% confidence intervals. Large states dominate the counts. "
                       "The chart on the right compares states by percentage change instead.")

        with c2:
            st.markdown("**Does poverty amplify heat's effect?**")
            curve = sensitivity_curve(model, (poverty_df["Poverty_Rate"].quantile(0.02),
                                              poverty_df["Poverty_Rate"].quantile(0.98)), pct)
            band = alt.Chart(curve).mark_area(opacity=0.2, color=color).encode(
                x=alt.X("Poverty_Rate:Q", title="State poverty rate (%)", scale=alt.Scale(zero=False)),
                y=alt.Y("Low:Q", title=f"% change in {outcome.lower()}"), y2="High:Q")
            line = alt.Chart(curve).mark_line(color=color).encode(x="Poverty_Rate:Q", y="Pct_Change:Q")
            dots = alt.Chart(sim).mark_circle(size=45, color="#495057").encode(
                x="Poverty_Rate:Q", y="Pct_Change:Q",
                tooltip=["State", alt.Tooltip("Poverty_Rate:Q", format=".1f"),
                         alt.Tooltip("Pct_Change:Q", title="% change", format="+.1f")])
            st.altair_chart((band + line + dots).properties(height=360), width="stretch")
            p_int = model.result.pvalues["log_heat:poverty_c"]
            verdict = ("a statistically significant amplifier" if p_int < 0.05
                       else "not a statistically significant amplifier")
            st.caption(f"Line: % change at a +{pct}% heat increase for a state with that poverty rate "
                       f"(shaded: 95% CI). Dots: individual states. Dots near zero are states with very few heat "
                       f"days, where a percentage increase adds little. For {outcome.lower()}, poverty is "
                       f"{verdict} (p = {p_int:.3f}).")

        with st.expander("Projection table"):
            st.dataframe(sim[["State", "Heat_Days", "Poverty_Rate", "Baseline", "Projected", "Extra",
                              "Extra_low", "Extra_high", "Pct_Change"]], hide_index=True, column_config={
                "Heat_Days": st.column_config.NumberColumn("Heat days/county/yr", format="%.0f"),
                "Poverty_Rate": st.column_config.NumberColumn("Poverty %", format="%.1f"),
                "Baseline": st.column_config.NumberColumn(format="%.0f"),
                "Projected": st.column_config.NumberColumn(format="%.0f"),
                "Extra": st.column_config.NumberColumn(format="%.0f"),
                "Extra_low": st.column_config.NumberColumn("95% low", format="%.0f"),
                "Extra_high": st.column_config.NumberColumn("95% high", format="%.0f"),
                "Pct_Change": st.column_config.NumberColumn("% change", format="%+.1f"),
            })

    with st.expander("How the model works"):
        b = model.result.params
        st.markdown(f"""
For each state and year, the model predicts the **{outcome.lower()} rate** from:

- **Heat:** log(1 + extreme heat days per county). A coefficient of **{b['log_heat']:.3f}** means about
  **{(1.10 ** b['log_heat'] - 1) * 100:.1f}% more {outcome.lower()} for every 10% more heat days**
  in a state with average poverty.
- **Poverty rate** from the Census Bureau, and its **interaction with heat**. A positive interaction
  means heat hits harder where poverty is higher.
- **State fixed effects** absorb permanent differences between states, such as climate,
  hospital coding practices and air-conditioning prevalence.
- **Year fixed effects** absorb shocks that hit every state in the same year, such as coding
  changes or COVID-19.

It is a quasi-Poisson regression on counts, with population as an exposure offset. Standard errors are
clustered by state. The effect of heat is estimated only from **within-state, year-to-year swings**,
so it isn't confused with Arizona simply being hotter than Maine.
""")
        st.dataframe(model.coef_table(), hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.4f") for c in ["Estimate", "CI low", "CI high", "p-value"]})
        st.caption("The poverty main effect compares a state with itself over time (for example, "
                   "recession years against boom years). Read it with caution, not as \"poverty protects\".")

# ================================================================ AI ANALYST
def build_ai_context() -> str:
    by_year = panel.groupby("Year").agg(Count=("Count", "sum"), Population=("Population", "sum"),
                                        Heat_Days_per_County=("Heat_Days", "mean"),
                                        States_Reporting=("State", "nunique")).reset_index()
    by_year["Rate_per_100k"] = by_year["Count"] / by_year.pop("Population") * 1e5
    peak, low = by_year.loc[by_year["Count"].idxmax()], by_year.loc[by_year["Count"].idxmin()]
    peak_rate = by_year.loc[by_year["Rate_per_100k"].idxmax()]
    n = min(3, len(by_year))
    first, last = by_year.head(n), by_year.tail(n)
    by_st = panel.groupby("State").agg(Count=("Count", "sum"), Heat_Days_per_County=("Heat_Days", "mean"),
                                       Avg_Rate_per_100k=("Rate_per_100k", "mean"),
                                       Avg_Poverty=("Poverty_Rate", "mean"),
                                       Years_Reported=("Year", "nunique")).reset_index()
    model = get_model(outcome)
    demo = ""
    if has_demographics:
        demo_t = filtered.groupby(["Age_Group", "Sex"])["Count"].sum().unstack()
        demo = f"\nBreakdown by age group and sex:\n{demo_t.to_csv()}"
    return f"""Current dashboard selection
- Outcome: {outcome}
- States: {', '.join(selected_states)}
- Years: {years[0]}-{years[1]}
- Age groups: {', '.join(ages) if has_demographics else 'n/a'}; Sex: {', '.join(sexes) if has_demographics else 'n/a'}

Pre-computed facts (use these rather than re-deriving them from the tables):
- Highest yearly count: {peak['Count']:,.0f} in {peak['Year']:.0f} ({peak['States_Reporting']:.0f} states reporting)
- Lowest yearly count: {low['Count']:,.0f} in {low['Year']:.0f} ({low['States_Reporting']:.0f} states reporting)
- Highest yearly rate per 100k: {peak_rate['Rate_per_100k']:.2f} in {peak_rate['Year']:.0f}
- Average rate per 100k, first {n} years ({first['Year'].min():.0f}-{first['Year'].max():.0f}): {first['Rate_per_100k'].mean():.2f}; last {n} years ({last['Year'].min():.0f}-{last['Year'].max():.0f}): {last['Rate_per_100k'].mean():.2f}
- States reporting went from {by_year['States_Reporting'].iloc[0]} to {by_year['States_Reporting'].iloc[-1]}, so raw counts partly reflect coverage; rates per 100k correct for this.
- Heat_Days_per_County = extreme heat days the average county in a state had that year.

Totals by year:
{by_year.round(2).to_csv(index=False)}
Totals by state (Avg_Rate_per_100k = average of yearly rates):
{by_st.round(2).to_csv(index=False)}{demo}
Model coefficients for {outcome} (quasi-Poisson, state and year fixed effects, population offset):
{model.coef_table().round(4).to_csv(index=False)}"""


def get_setting(name: str, default: str = "") -> str:
    """Streamlit secrets (Community Cloud) first, then environment / .env."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml at all
        pass
    return os.environ.get(name, default)


AI_SESSION_LIMIT = int(get_setting("AI_SESSION_LIMIT", "10"))  # questions per visitor session
AI_DAILY_LIMIT = int(get_setting("AI_DAILY_LIMIT", "200"))     # questions per day, all visitors
AI_HISTORY_TURNS = 6                                            # past messages sent with each question


@st.cache_resource
def ai_usage() -> dict:
    """App-wide question counter, shared by every visitor of this server process."""
    return {"day": None, "count": 0}


def daily_remaining() -> int:
    usage, today = ai_usage(), pd.Timestamp.now(tz="UTC").date()
    if usage["day"] != today:
        usage.update(day=today, count=0)
    return AI_DAILY_LIMIT - usage["count"]


def queue_suggestion():
    # Send a clicked suggestion once, then clear the pill so reruns don't resend it
    st.session_state.pending_prompt = st.session_state.suggestion
    st.session_state.suggestion = None


with tab_ai:
    st.markdown("Ask questions about the data **currently selected in the sidebar**. "
                "The assistant only sees the summary tables below.")
    api_key = get_setting("GROQ_API_KEY")
    groq_model = get_setting("GROQ_MODEL", "openai/gpt-oss-120b")
    context = build_ai_context()
    with st.expander("What the assistant can see"):
        st.code(context, language=None)

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("ai_questions", 0)
    session_left = AI_SESSION_LIMIT - st.session_state.ai_questions
    chat_enabled = bool(api_key) and session_left > 0 and daily_remaining() > 0

    if not api_key:
        st.info("The AI Analyst is turned off because no Groq API key is configured. Everything else in the "
                "dashboard works without it. To turn it on, add `GROQ_API_KEY` to `.env` (local) or to the "
                "app's Secrets (Streamlit Community Cloud).", icon="🔑")
    elif not chat_enabled:
        reason = ("You've used all the questions for this session." if session_left <= 0
                  else "The dashboard has reached its shared daily limit for AI questions.")
        st.warning(f"{reason} The rest of the dashboard still works. Check back later, or run the app "
                   "locally with your own free Groq key (see the README).", icon="⏳")
    else:
        st.caption(f"{session_left} of {AI_SESSION_LIMIT} questions left in this session.")

    st.pills("Try asking", [
        "Which state has the highest rate, and how does its poverty compare?",
        "How have cases changed since the early 2000s?",
        "Explain the model results in plain language.",
    ], key="suggestion", on_change=queue_suggestion, disabled=not chat_enabled)
    if st.button("Clear chat", type="tertiary"):
        st.session_state.messages = []

    chat = st.container()
    for m in st.session_state.messages:
        chat.chat_message(m["role"]).markdown(m["content"])

    prompt = (st.chat_input("Ask about heat and health in your selection…", max_chars=500,
                            disabled=not chat_enabled)
              or st.session_state.pop("pending_prompt", None))
    if prompt and chat_enabled:
        chat.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.ai_questions += 1
        ai_usage()["count"] += 1
        system = ("You are a careful public-health data analyst inside a dashboard about extreme heat and "
                  "heat-related health outcomes in US states (CDC Environmental Public Health Tracking, "
                  "Census SAIPE poverty). Answer ONLY from the data below. If the answer is not in the "
                  "data, say so plainly; never invent numbers. Quote the figures you use in plain text (no citation brackets). Keep answers "
                  "concise. Remember: states report different years, so raw yearly totals partly reflect "
                  "reporting coverage. Prefer rates per 100k when comparing.\n\n" + context)
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages[-AI_HISTORY_TURNS:]]
        try:
            with chat, st.spinner("Thinking…"):
                reply = groq.Groq(api_key=api_key).chat.completions.create(
                    model=groq_model, temperature=0.2,
                    messages=[{"role": "system", "content": system}] + history)
            answer = reply.choices[0].message.content
        except Exception as e:
            answer = f"Error contacting Groq ({groq_model}): {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()  # redraw with the answer and the updated question count

# ================================================================ NYC
with tab_nyc:
    st.markdown("A closer look at one city. NYC has dense housing, little tree canopy in places, and sharp "
                "income gaps between neighborhoods, which makes **where** heat lands very uneven. *This tab is a fixed 2022 snapshot of New York City, so the sidebar filters (states, years, age, sex) don't change it.*")
    nyc = pd.read_csv("nyc/nyc_heat_er_visits_2022.csv")
    citywide = nyc.loc[nyc["GeoTypeDesc"] == "Citywide", "Age-adjusted rate per 100,000"].iloc[0]
    boroughs = nyc[nyc["GeoTypeDesc"] == "Borough"].rename(columns={
        "Age-adjusted rate per 100,000": "Rate", "Number": "Visits"})

    n1, n2 = st.columns([2, 3])
    with n1:
        st.markdown("**Heat-stress ER visits, 2022 (age-adjusted per 100k)**")
        bars = alt.Chart(boroughs).mark_bar(color=OUTCOME_COLORS["ER Visits"]).encode(
            x=alt.X("Rate:Q", title="Per 100k residents"),
            y=alt.Y("Borough:N", sort="-x", title=None),
            tooltip=["Borough", "Visits", alt.Tooltip("Rate:Q", format=".1f")])
        rule = alt.Chart(pd.DataFrame({"x": [citywide]})).mark_rule(strokeDash=[4, 3], color="#868e96").encode(x="x:Q")
        st.altair_chart((bars + rule).properties(height=260), width="stretch")
        top = boroughs.nlargest(1, "Rate").iloc[0]
        st.caption(f"Dashed line: citywide rate ({citywide}). {top['Borough']} is highest at {top['Rate']}, "
                   f"{top['Rate'] / citywide:.1f}× the city average. Source: NYC Environment & Health Data Portal.")
    with n2:
        st.image("nyc/Map2_Health_Outcomes.jpg", width="stretch",
                 caption="Borough ER visit rates over land surface temperature (QGIS).")
    st.image("nyc/Map1_Heat_Exposure.jpg", width="stretch",
             caption="Land surface temperature from Landsat 8 (June 19, 2022) with tree canopy and ZIP code boundaries. "
                     "Hot spots line up with dense, low-canopy neighborhoods.")

# ================================================================ METHODS
with tab_methods:
    st.markdown("""
#### Data
| Source | What | Level |
|---|---|---|
| CDC Environmental Public Health Tracking | Extreme heat days per county (NOAA temperature data) | County-year, averaged to state |
| CDC Environmental Public Health Tracking | Heat-related hospitalizations and ER visits, by age and sex | State-year |
| CDC Environmental Public Health Tracking | Heat-related deaths (suppressed when < 10) | State-year |
| U.S. Census Bureau SAIPE | Poverty rate, all ages, and population | State-year |
| NYC Environment & Health Data Portal, Landsat 8 | Borough ER visit rates, land surface temperature | Borough, 30 m raster |

**About the heat measure.** The CDC extract calls the column "extreme heat events", but its values reach
194 per county per year. That's impossible for multi-day events, so they are counts of **heat days**. The
dashboard averages them across counties: a state's value is how many extreme heat days its typical county
had. Summing instead would count one heat wave once for every county it touched, giving totals like
"80,000 events" in a year.

Health outcomes are **heat-related** illness only (ICD-coded heat illness), not all hospitalizations.

Only states that take part in CDC tracking are included (31 for heat and hospitalizations,
30 for ER visits). Results may not generalize to the rest of the country.

#### Model
`log E[count] = log(population) + β₁·log(1+heat) + β₂·poverty + β₃·log(1+heat)·poverty + state FE + year FE`

This is a quasi-Poisson GLM with standard errors clustered by state, fit separately for each outcome. The
simulator scales each state's recent average heat days and applies the fitted rate ratio to its
recent average count.

#### Limitations
- **Association, not proof.** Fixed effects remove a lot of confounding, but not all of it.
- **No lag structure.** This uses annual data, so it can't separate heat waves from cumulative heat.
- **Reporting varies.** States joined the tracking network in different years, so use the rates.
- **Air conditioning** isn't published by state and year, so it's absorbed by the state fixed
  effects rather than modeled directly.
- **Heat scenarios are hypothetical.** The simulator varies event frequency, not temperature.
""")
