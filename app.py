import streamlit as st
import pandas as pd
import os
import io
import groq
from dotenv import load_dotenv

load_dotenv()
from PIL import Image
from ml_model import load_and_merge_data, add_socioeconomic_indicators, build_advanced_model, simulate_climate_scenarios

# Setting page config to be wide and clean for a modern dashboard look
st.set_page_config(page_title="Heat's Toll on Health", layout="wide")

st.markdown("""
<style>
    .reportview-container {
        background: #FAFAFA
    }
    .sidebar .sidebar-content {
        background: #F0F2F6
    }
</style>
""", unsafe_allow_html=True)

st.title("🔥 Heat's Toll on Health Dashboard")
st.markdown("Exploring the relationship between Extreme Heat Events, Health Outcomes, and Socioeconomic Vulnerabilities.")

@st.cache_data
def load_app_data():
    df = load_and_merge_data("./EDAV-Project/data_clean/")
    df = add_socioeconomic_indicators(df)
    model = build_advanced_model(df)
    return df, model

df, model = load_app_data()

# Data Summary for LLM Context
data_summary = f"""
The dataset contains state-year observations from 2000-2022.
Columns include: State, Year, EHE (Extreme Heat Events), Hosps (Hospitalizations), 
ER.Visits (ER Visits), Deaths, Poverty_Rate, Pct_No_AC.
Total rows: {len(df)}. Total states: {len(df['State'].unique())}.
Latest year data:
{df[df['Year'] == 2022].head().to_string()}
"""

# Configure Groq (Ensure API key is set in environment)
api_key = os.environ.get("GROQ_API_KEY", "")

tab1, tab2, tab3 = st.tabs(["📊 Data Visualizations", "🧠 ML Simulation Engine", "🤖 AI Assistant"])

with tab1:
    st.header("Baseline Insights from the CDC")
    st.markdown("These visualizations were rendered natively in R based on the underlying EPH trajectory.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Surge in Health Impacts (Since 2000)")
        try:
            img1 = Image.open("plots/plot1_trends.png")
            st.image(img1, use_container_width=True)
        except Exception:
            st.warning("plot1_trends.png not generated.")

    with col2:
        st.subheader("Decade-Long Intensification of Heat Events")
        try:
            img3 = Image.open("plots/plot3_heatmap.png")
            st.image(img3, use_container_width=True)
        except Exception:
            st.warning("plot3_heatmap.png not generated.")
            
    st.subheader("Standardized Heat Events vs Hospitalizations")
    try:
        img2 = Image.open("plots/plot2_standardized.png")
        st.image(img2, use_container_width=True)
    except Exception:
        st.warning("plot2_standardized.png not generated.")

with tab2:
    st.header("Climate Consequence Simulator")
    st.markdown("Adjust the slider to simulate the health impact of increased extreme heat events across US States. The underlying **Distributed Lag Non-Linear Model** accounts for J-Curve temperature trajectories and poverty interactions.")
    
    temp_increase = st.select_slider(
        "Select Temperature Increase Scenario",
        options=["Baseline", "+1°C (approx +20% Heat Events)", "+2°C (approx +50% Heat Events)", "+3°C (approx +100% Heat Events)"]
    )
    
    multiplier = 1.0
    if "+1°C" in temp_increase:
        multiplier = 1.2
    elif "+2°C" in temp_increase:
        multiplier = 1.5
    elif "+3°C" in temp_increase:
        multiplier = 2.0
        
    if multiplier > 1.0:
        sim_results = simulate_climate_scenarios(df, model, increase_factor=multiplier)
        recent_year = sim_results['Year'].max()
        sim_recent = sim_results[sim_results['Year'] == recent_year]
        
        st.write(f"### Projected Additional Hospitalizations for {recent_year} under {temp_increase}")
        display_df = sim_recent[['State', 'Original_EHE', 'Baseline_Hosps', 'Simulated_Hosps', 'Diff']].copy()
        display_df['Diff'] = display_df['Diff'].astype(int)
        display_df['Simulated_Hosps'] = display_df['Simulated_Hosps'].astype(int)
        display_df['Baseline_Hosps'] = display_df['Baseline_Hosps'].astype(int)
        
        st.dataframe(display_df.sort_values(by='Diff', ascending=False).head(15), use_container_width=True)
        st.markdown("**Interaction Check: Do states with higher poverty suffer worse marginal impacts?**")
        st.scatter_chart(data=sim_recent, x='Poverty_Rate', y='Diff', color='State')
    else:
        st.info("Move the slider to simulate future climate scenarios.")

with tab3:
    st.header("Agentic AI Data Analyst")
    st.markdown("Ask natural language questions about the underlying CDC heat impacts dataset!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("E.g: 'Which state had the highest hospitalizations in 2022?'"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if not api_key:
            response = "⚠️ Please provide a Groq API Key in the `.env` file to use the AI Assistant."
        else:
            try:
                client = groq.Groq(api_key=api_key)
                system_prompt = f"""
                You are a data analyst assistant for a Heat-Health dashboard.
                You have access to a dataset containing Heat Events, Hospitalizations, ER Visits, and Deaths for US States from 2000 to 2022.
                Here is a summary of the dataset:
                {data_summary}
                
                Please answer the user's question accurately based on this information. 
                If you need to guess values or infer trends based on general US knowledge and the schema, you may do so carefully.
                """
                
                groq_messages = [{"role": "system", "content": system_prompt}]
                for m in st.session_state.messages:
                    if m["role"] == "user":
                        groq_messages.append({"role": "user", "content": m["content"]})
                    elif m["role"] == "assistant":
                        groq_messages.append({"role": "assistant", "content": m["content"]})
                        
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=groq_messages,
                    temperature=0.3,
                )
                response = completion.choices[0].message.content
            except Exception as e:
                response = f"Error communicating with AI: {str(e)}"
                
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
