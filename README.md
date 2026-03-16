# Heat Toll on Health - Interactive Dashboard

This repository contains an interactive [Streamlit](https://streamlit.io/) dashboard that visualizes the CDC Environmental Public Health (EPH) dataset, simulating the toll of Extreme Heat Events on hospitalizations across the United States.

It combines pre-rendered R visualizations (`ggplot2`) with an advanced Python Machine Learning model (Distributed Lag Non-Linear Model equivalent via polynomial splines) that estimates local county/state vulnerabilities when factoring in socioeconomic metrics like poverty and lack of AC.

Finally, it integrates the **Groq AI (Llama 3)** model so that users can chat directly with an Agentic AI Data Analyst about the dataset.

## Setup & Installation

**Prerequisites:**
- R and `Rscript` installed (for regenerating plots)
- Python 3.9+
- A valid [Groq API Key](https://console.groq.com/) for the AI Agent

### 1. Environment Setup

Clone this repository and create a Python virtual environment:

```bash
git clone https://github.com/ka1899/Heat_Toll_on_Health.git
cd Heat_Toll_on_Health

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required Python packages
pip install pandas numpy statsmodels scikit-learn streamlit matplotlib seaborn groq python-dotenv
```

### 2. Configure the LLM Agent

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_api_key_here
```

### 3. Generate Visualizations (Optional)
If you want to regenerate the baseline R plots from the curated CDC data, you can run the R script:

```bash
Rscript generate_plots.R
```
*(This will populate the `plots/` folder with updated PNGs used in the Streamlit UI).*

### 4. Run the Dashboard

To launch the dashboard locally:

```bash
streamlit run app.py
```
This will open up a local web server (usually at `http://localhost:8501`).

## Project Structure
- `app.py`: Main Streamlit application containing UI logic, tabs, simulation interactive elements, and Groq SDK wiring.
- `ml_model.py`: Handles data pipeline engineering, simulation projection logic, and fits the underlying statistical model (OLS with polynomial temperature curves and socioeconomic fixed effects).
- `generate_plots.R`: Standalone R script built that renders base state heatmaps, indexes, and trends to PNGs.
- `EDAV-Project/`: Base directory containing the clean CDC CSV extracts (`data_clean/`).
- `plots/`: Output directory where R visualizations reside.
