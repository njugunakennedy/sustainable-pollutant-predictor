import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import os

# Set random seed for reproducibility
np.random.seed(42)

@st.cache_data
def synthesize_data():
    """Synthesizes a realistic dataset for Milk Fat Removal based on pH, 
    matching the trends reported in the provided research paper (Fig. 2).
    """
    # Reported data trend: Max removal at pH 4, drops significantly at pH 2 and pH 10.
    ph_values = np.linspace(2.0, 10.0, 100)
    
    # Model the non-linear trend (parabolic shape, peaking near pH 4)
    # A quadratic function that peaks near 4 and decreases towards 2 and 10.
    # The constants are adjusted slightly to hit ~100% at pH 4 and drop to ~30% at pH 2/10.
    a = 3.5  
    
    # We will use the formula: Efficiency = Max_Eff - a * (pH - 4)^2 + noise
    removal_ideal = 100 - a * (ph_values - 4)**2
    
    # Add noise to simulate real experimental results
    noise = np.random.normal(0, 5, 100)
    # Clip results to physical limits (20% to 100% based on paper's Fig. 2 values)
    removal_eff = np.clip(removal_ideal + noise, 20, 100) 
    
    df = pd.DataFrame({
        'pH': ph_values,
        'Milk_Fat_Removal_Efficiency (%)': removal_eff
    })
    
    return df

@st.cache_resource
def train_prediction_model():
    """Trains a Polynomial Regression model to fit the non-linear pH vs. removal curve."""
    # Get cached data
    df = synthesize_data()
    X = df[['pH']]
    y = df['Milk_Fat_Removal_Efficiency (%)']
    
    # Use a 3rd degree polynomial to capture the curve shape (peak and drop-off)
    model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    model.fit(X, y)
    return model

def predict_removal(model, ph_value):
    """Uses the trained model to predict removal efficiency for a single pH value."""
    # Ensure input is a DataFrame/array format the model expects
    ph_input = np.array([[ph_value]])
    prediction = model.predict(ph_input)[0]
    return np.clip(prediction, 0, 100) # Ensure prediction stays within 0-100%

# --- Streamlit UI and Visualization ---

# Configure page settings
st.set_page_config(
    page_title="Sustainable Wastewater Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and train data (cached for performance)
data_df = synthesize_data()
ml_model = train_prediction_model()

# --- HEADER AND INTRO ---
st.title("🧪 Sustainable Pollutant Removal Predictor")
st.markdown("### Predicting Milk Fat Removal using Biochar and Machine Learning")

# Research Purpose and Sustainability Impact Section
with st.expander("🎯 Research Objectives & Sustainability Impact", expanded=True):
    st.markdown("""
    ### Research Purpose
    
    This research addresses a critical sustainability challenge in the dairy industry by:
    
    1. **Fabricating biochar from milk sludge** for the adsorption of organic matter (OM) and nutrients from milk processing wastewater
    2. **Conducting batch adsorption experiments** to evaluate biochar performance in removing pollutants and characterizing biochar properties
    3. **Proposing industrial-scale applications** via simulation to enhance wastewater treatment sustainability in the dairy sector
    
    ### 🌱 Sustainability Solution: Circular Economy Approach
    
    **The Problem:** Milk processing plants generate massive amounts of wastewater and sludge that require costly treatment and disposal.
    
    **The Solution:** Convert waste sludge into valuable biochar adsorbent - transforming a waste product into a resource!
    
    - **Waste Input:** 6,144 kg of sludge per day (from 500,000 L milk production)
    - **Resource Output:** 1,947.65 kg of biochar per day (32% yield)
    - **Impact:** Biochar removes pollutants from 1,236,000 liters of wastewater daily
    
    This creates a **circular economy** where waste becomes a resource, reducing both disposal costs and treatment chemical requirements.
    """)
    
    # Industrial Scale Case Study
    st.markdown("---")
    st.subheader("🏭 Industrial Scale Application: Brookside Milk Plant, Kenya")
    
    col_case1, col_case2 = st.columns(2)
    with col_case1:
        st.metric("Daily Milk Production", "500,000 L")
        st.metric("Wastewater Generated", "1,236,000 L/day")
    with col_case2:
        st.metric("Sludge Produced", "6,144 kg/day")
        st.metric("Biochar Yield", "1,947.65 kg/day")
    
    st.markdown("""
    The research simulated a **two-tank system in series** to handle industrial-scale milk fat COD and anionic pollutant removal, 
    demonstrating the viability of this sustainable approach for real-world dairy processing facilities.
    """)

st.markdown("---")
st.markdown("""
This application demonstrates **Polynomial Regression** to predict pollutant removal efficiency based on key operational parameters, 
leveraging data trends from the paper: *Pollutant Removal and Nutrient Recovery from Milk Processing Wastewater*.

The paper identified **pH** as the **only significant factor** for Milk Fat removal efficiency.
""")

st.markdown("---")


# --- SIDEBAR (ML INTERACTION) ---
st.sidebar.header("🔬 Predictive Model Control")
st.sidebar.markdown("**Predict Milk Fat Removal Efficiency**")

# Interactive pH slider
selected_ph = st.sidebar.slider(
    'Adjust Solution pH:',
    min_value=2.0,
    max_value=10.0,
    value=4.0,
    step=0.1,
    help="pH is the critical factor for Milk Fat Removal (ANOVA result)."
)

# Prediction
predicted_removal = predict_removal(ml_model, selected_ph)

st.sidebar.subheader("Predicted Removal")
st.sidebar.metric(
    label=f"Milk Fat Removal @ pH {selected_ph:.1f}",
    value=f"{predicted_removal:.2f}%",
    delta_color="off"
)

st.sidebar.markdown("---")
st.sidebar.header("Key Findings")
st.sidebar.metric("Maximum Adsorption Capacity ($Q_{\\text{max}}$)", "217.4 mg/g")
st.sidebar.metric("Equilibrium Time", "60 minutes (Fast Adsorption)")


# --- MAIN CONTENT LAYOUT ---
col1, col2 = st.columns([3, 2])

with col1:
    st.header("1. ML Prediction Curve and Data Visualization")
    st.markdown("The Polynomial Regression model (Degree 3) is used to fit the highly non-linear effect of pH on Milk Fat Removal reported in the study.")
    
    # Generate prediction line data
    ph_range = np.linspace(2.0, 10.0, 50).reshape(-1, 1)
    predicted_curve = ml_model.predict(ph_range)
    
    prediction_df = pd.DataFrame({
        'pH': ph_range.flatten(),
        'Prediction (%)': predicted_curve
    })
    
    # Plotly visualization
    fig = px.scatter(
        data_df, 
        x='pH', 
        y='Milk_Fat_Removal_Efficiency (%)', 
        title='pH vs. Milk Fat Removal Efficiency (Synthetic Data)',
        color_discrete_sequence=['#00B4D8'] # Secondary Cyan
    )
    
    # Add prediction line
    fig.add_scatter(
        x=prediction_df['pH'], 
        y=prediction_df['Prediction (%)'], 
        mode='lines', 
        name='ML Prediction Curve',
        line=dict(color='#0077B6', width=4) # Primary Blue
    )

    # Highlight the user's current selection
    fig.add_scatter(
        x=[selected_ph],
        y=[predicted_removal],
        mode='markers',
        marker=dict(size=12, color='#F94144', symbol='star'), # Energetic Red
        name='Current Prediction'
    )
    
    fig.update_layout(
        xaxis_title="Solution pH",
        yaxis_title="Removal Efficiency (%)",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("2. Critical Factors & Conclusion")
    
    st.markdown("---")
    st.subheader("BBD Optimization Results")
    st.markdown("""
    The Box-Behnken Design (BBD) analysis revealed that different pollutants are controlled by different factors:
    """)
    
    col2a, col2b = st.columns(2)
    with col2a:
        st.info("**Milk Fat Removal**")
        st.markdown("**Significant Factor:** $\\text{pH}$")
        st.markdown("**Optimum pH:** 4")
        st.markdown("**Mechanism:** Chemisorption / Hydrophobic Interactions")
        
    with col2b:
        st.info("**Methyl Orange Removal**")
        st.markdown("**Significant Factor:** Adsorbent **Dosage**")
        st.markdown("**Optimum Efficiency:** 57% (Low pH 2 increases efficiency by 40%)")
        st.markdown("**Mechanism:** Physisorption / Electrostatic Interactions")

    st.markdown("---")
    st.subheader("Conclusion")
    st.markdown(
        """
        The project confirms that biochar derived from milk sludge is a **sustainable and effective** adsorbent, particularly for Milk Fat (Organic Matter). The two-tank industrial simulation confirmed the potential to handle large wastewater volumes, demonstrating the viability of this circular economy approach.
        """
    )

# --- INDUSTRIAL SCALE SIMULATION SECTION ---
st.markdown("---")
st.header("3. Industrial Scale Two-Tank System Design")

st.markdown("""
The research developed a **two-tank system in series** to simulate industrial-scale treatment of milk processing wastewater. 
This design enables continuous treatment of large volumes while optimizing biochar utilization and pollutant removal efficiency.
""")

# Display the two-tank scenario images
img_col1, img_col2 = st.columns(2)

with img_col1:
    partial_img_path = "Images/Scenario Partail.png"
    if os.path.exists(partial_img_path):
        st.image(partial_img_path, caption="Partial Treatment Scenario", use_container_width=True)
    else:
        st.warning("Scenario Partial image not found")

with img_col2:
    whole_img_path = "Images/Scenario Whole.png"
    if os.path.exists(whole_img_path):
        st.image(whole_img_path, caption="Complete Treatment Scenario", use_container_width=True)
    else:
        st.warning("Scenario Whole image not found")

st.markdown("""
**Key Design Features:**
- **Tank 1:** Primary treatment for high-concentration pollutants
- **Tank 2:** Secondary treatment for polishing and final removal
- **Continuous Flow:** Handles 1,236,000 L/day wastewater volume
- **Biochar Regeneration:** Enables reuse of adsorbent material
- **Scalability:** Design validated for Brookside milk plant case study
""")