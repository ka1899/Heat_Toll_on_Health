# Heat's Toll on Health: Interactive Dashboard

An interactive [Streamlit](https://streamlit.io/) dashboard on how extreme heat days relate to
heat-related hospitalizations, ER visits and deaths across U.S. states (2000–2022), and whether
poverty makes those effects worse.

**What's inside**

| Tab | What it does |
|---|---|
| 📊 **Overview** | Key findings in plain English (computed from the data and model, not hard-coded), headline metrics, heat and health trends, a state map, a state × year heatmap, age/sex breakdown and heat-vs-health scatter. Everything responds to the sidebar filters, and the filtered data can be downloaded as CSV. |
| 🌡️ **Heat Scenario Simulator** | Increase extreme heat days by 0–200% and see projected extra cases per state with 95% confidence intervals, plus how the effect changes with a state's poverty rate. |
| 🤖 **AI Analyst** | Chat with an LLM (via Groq) that sees summary tables of your current filter selection and is told not to guess. Questions are rate-limited per visitor. |
| 🗽 **NYC Spotlight** | A fixed 2022 snapshot (not affected by the sidebar): borough-level heat-stress ER visits alongside Landsat land-surface-temperature and tree-canopy maps. |
| 📖 **Methods** | Data sources, model specification and limitations. |

**Sidebar filters (slicers):** health outcome · states · year range · age group · sex.

## Quick start

Requires Python 3.9+.

```bash
git clone https://github.com/ka1899/Heat_Toll_on_Health.git
cd Heat_Toll_on_Health

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then add your Groq API key (only needed for the AI tab)
streamlit run app.py
```

The app opens at http://localhost:8501. Everything except the AI Analyst works without an API key.
The map tiles load from a CDN, so you need an internet connection.

Run the tests with `pip install -r requirements-dev.txt && pytest`.

## Deploy (Streamlit Community Cloud)

The app runs as-is on [Streamlit Community Cloud](https://share.streamlit.io) (free):

1. Sign in at share.streamlit.io with GitHub and choose **Create app**.
2. Pick this repository, branch `main`, main file `app.py`. Under **Advanced settings**, choose
   Python 3.12 or newer.
3. Optional, for the AI tab: in **Secrets**, paste the contents of
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) with your real Groq key.
   Secrets can also be added later under the app's **Settings → Secrets**.
4. Deploy. Dependencies install from `requirements.txt`.

**Protecting your Groq quota.** On a public URL anyone can use the AI tab, so it is capped:
each visitor session gets `AI_SESSION_LIMIT` questions (default 10) and the whole app gets
`AI_DAILY_LIMIT` per day (default 200, resets at midnight UTC or when the app restarts). Set either
in Secrets to change them. Without a key the AI tab shows a notice and everything else works.
Never commit `.streamlit/secrets.toml` or `.env`; both are gitignored.

## Data

| Source | Variables | Level |
|---|---|---|
| [CDC Environmental Public Health Tracking](https://ephtracking.cdc.gov/) | Extreme heat days per county (NOAA); hospitalizations and ER visits for heat-related illness by age and sex; heat-related deaths | County-year (heat), state-year (health) |
| [U.S. Census Bureau SAIPE](https://www.census.gov/programs-surveys/saipe.html) | Poverty rate (all ages), population | State-year |
| [NYC Environment & Health Data Portal](https://a816-dohbesp.nyc.gov/IndicatorPublic/), USGS Landsat 8 | Borough heat-stress ER visit rates (2022); land surface temperature (June 19, 2022) | Borough / 30 m raster |

**About the heat measure:** the CDC extract labels the column `EHE` ("extreme heat events"), but values
reach 194 per county per year. That's impossible for multi-day events, so they are counts of **heat days**.
States are given the **average across their counties**: how many extreme heat days a typical county had.
Summing across counties would count one heat wave once per county it touched.

The cleaned CDC extracts are in `EDAV-Project/data_clean/`. The poverty table `data/state_poverty.csv` is
built from the Census Bureau's public SAIPE files by:

```bash
python scripts/build_poverty.py
```

## Model

For each outcome, a quasi-Poisson regression on state-year counts:

```
log E[count] = log(population)
             + β₁·log(1 + heat days per county)
             + β₂·poverty rate
             + β₃·log(1 + heat days per county)·poverty rate
             + state fixed effects + year fixed effects
```

- **State fixed effects** absorb permanent differences between states, such as climate, hospital
  coding and air-conditioning prevalence. **Year fixed effects** absorb nationwide shocks.
  The heat effect is therefore estimated from within-state, year-to-year swings in heat.
- Standard errors are clustered by state.
- Because heat enters on a log scale, raising heat days by *x*% multiplies expected cases by
  about (1 + *x*)^(β₁ + β₃·poverty). The effect rises with diminishing returns, and is steeper in
  higher-poverty states when β₃ > 0.

Current estimates (`python ml_model.py` prints these):

| Outcome | Heat (β₁) | Heat × poverty (β₃) | Reading |
|---|---|---|---|
| Hospitalizations | 0.322 (p < 0.001) | 0.005 (p = 0.40) | +10% heat days → ≈ +3.1% hospitalizations |
| ER visits | 0.201 (p < 0.001) | 0.001 (p = 0.87) | +10% → ≈ +1.9% |
| Deaths | 0.439 (p < 0.001) | 0.012 (p = 0.70) | +10% → ≈ +4.3% |

Heat has a clear, significant effect on all three outcomes. The data **does not** show poverty
amplifying that effect at the state level. State-wide poverty averages may be too coarse to show it;
neighborhood-level analyses like the NYC tab are better suited.

**Limitations:** these are associations, not causal proof. There is no lag structure, because the data
is annual. Only the 30–31 states in the CDC tracking network are covered. Deaths below 10 are suppressed.
Air conditioning isn't available by state and year, so the state fixed effects absorb it instead.

## Project structure

```
app.py                  Streamlit dashboard (filters, tabs, charts, Groq chat)
tests/                  AppTest smoke tests (pytest)
.streamlit/             Theme (config.toml) and secrets template
heat_data.py            Loads CDC + Census data into tidy, filterable tables
ml_model.py             Model fitting, scenario simulation, poverty sensitivity curve
scripts/build_poverty.py  Builds data/state_poverty.csv from Census SAIPE files
data/                   Census poverty + population by state-year
nyc/                    NYC maps (exported from QGIS) and borough ER data
EDAV-Project/           Original Quarto EDA project and cleaned CDC extracts
generate_plots.R        Optional: static ggplot2 figures for reports → plots/
```

To regenerate the static R figures (optional): `Rscript generate_plots.R` (needs `tidyverse` and `scales`).
