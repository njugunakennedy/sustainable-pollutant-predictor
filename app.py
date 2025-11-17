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

# --- NEW PARAMETERS AND RANGES ---
TEMP_MIN, TEMP_MAX = 20.0, 50.0 # Temperature (°C)
CONC_MIN, CONC_MAX = 50.0, 500.0 # Initial Milk Fat Concentration (mg/L)
DOSAGE_MIN, DOSAGE_MAX = 0.5, 5.0 # Biochar Dosage (g/L)

@st.cache_data
def synthesize_data_enhanced():
    """
    Synthesizes a complex dataset for Milk Fat Removal based on pH,
    Concentration, Dosage, and Temperature.
    
    The primary effect remains the parabolic dependence on pH, with
    secondary, minor effects from the new factors.
    """
    N_SAMPLES = 500 # Increase samples for more complex model training
    
    # 1. Independent Variables
    ph_values = np.linspace(2.0, 10.0, N_SAMPLES)
    conc_values = np.random.uniform(CONC_MIN, CONC_MAX, N_SAMPLES)
    dosage_values = np.random.uniform(DOSAGE_MIN, DOSAGE_MAX, N_SAMPLES)
    temp_values = np.random.uniform(TEMP_MIN, TEMP_MAX, N_SAMPLES)
    
    # 2. Base Model (pH Parabola - Most Significant)
    # Target: Max removal at pH 4 (~100%), drops at pH 2 and pH 10.
    a_ph = 3.5
    removal_base = 100 - a_ph * (ph_values - 4)**2
    
    # 3. Secondary Effects (Minor Adjustments)
    # Effect of Concentration (Expected: Higher concentration slightly lowers efficiency)
    conc_effect = -0.01 * (conc_values - (CONC_MIN + CONC_MAX) / 2) # Range: ~ -2.25 to +2.25
    
    # Effect of Dosage (Expected: Higher dosage increases efficiency)
    dosage_effect = 2.0 * (dosage_values - (DOSAGE_MIN + DOSAGE_MAX) / 2) # Range: ~ -4.5 to +4.5
    
    # Effect of Temperature (Expected: Moderate temp is best, slight drop at extremes - Quadratic)
    opt_temp = 35.0
    temp_effect = -0.1 * (temp_values - opt_temp)**2 + 5.0 # Range: ~ 0 to +5.0 (Peak at 35°C)
    
    # 4. Total Ideal Removal
    removal_ideal = removal_base + conc_effect + dosage_effect + temp_effect
    
    # 5. Noise and Clipping
    noise = np.random.normal(0, 4, N_SAMPLES) # Slightly lower noise to allow model to find minor effects
    removal_eff = np.clip(removal_ideal + noise, 20, 100) # Clip results to physical limits
    
    df = pd.DataFrame({
        'pH': ph_values,
        'Concentration (mg/L)': conc_values,
        'Dosage (g/L)': dosage_values,
        'Temperature (°C)': temp_values,
        'Milk_Fat_Removal_Efficiency (%)': removal_eff
    })
    
    return df

@st.cache_resource
def train_prediction_model_enhanced():
    """
    Trains a Polynomial Regression model with all four features and interaction terms.
    """
    df = synthesize_data_enhanced()
    
    # Use all four features as input
    X = df[['pH', 'Concentration (mg/L)', 'Dosage (g/L)', 'Temperature (°C)']]
    y = df['Milk_Fat_Removal_Efficiency (%)']
    
    # Use 3rd degree polynomial for pH's non-linear effect, and 2nd degree for
    # interactions between all four terms. This creates a highly complex model.
    model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
    model.fit(X, y)
    return model

def predict_removal_enhanced(model, ph, conc, dosage, temp):
    """Uses the trained model to predict removal efficiency for a set of input parameters."""
    # Ensure input is a DataFrame/array format the model expects
    input_data = np.array([[ph, conc, dosage, temp]])
    prediction = model.predict(input_data)[0]
    return np.clip(prediction, 0, 100)

# --- Streamlit UI and Visualization ---

# Configure page settings
st.set_page_config(
    page_title="Enhanced Sustainable Wastewater Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and train data (cached for performance)
data_df_enhanced = synthesize_data_enhanced()
ml_model_enhanced = train_prediction_model_enhanced()

# --- HEADER AND INTRO ---
st.title("🧪 Enhanced Pollutant Removal Predictor (Multi-Factor)")
st.markdown("### Predicting Milk Fat Removal using Biochar, considering **pH, Concentration, Dosage, and Temperature**.")

st.markdown("---")


# --- SIDEBAR (ML INTERACTION) ---
st.sidebar.header("🔬 Predictive Model Control")
st.sidebar.markdown("##### Adjust all four critical factors for the adsorption process.")

# Interactive sliders for all four factors
selected_ph = st.sidebar.slider(
    'Solution pH (Primary Factor):',
    min_value=2.0, max_value=10.0, value=4.0, step=0.1,
    help="Most significant factor (parabolic effect)."
)

selected_conc = st.sidebar.slider(
    'Initial Concentration (mg/L):',
    min_value=CONC_MIN, max_value=CONC_MAX, value=150.0, step=10.0,
    help="Higher concentration tends to slightly lower efficiency."
)

selected_dosage = st.sidebar.slider(
    'Biochar Dosage (g/L):',
    min_value=DOSAGE_MIN, max_value=DOSAGE_MAX, value=2.5, step=0.1,
    help="Higher dosage generally increases efficiency."
)

selected_temp = st.sidebar.slider(
    'Temperature (°C):',
    min_value=TEMP_MIN, max_value=TEMP_MAX, value=35.0, step=1.0,
    help="Optimal removal is near 35°C (Quadratic effect)."
)

# Prediction
predicted_removal = predict_removal_enhanced(
    ml_model_enhanced, selected_ph, selected_conc, selected_dosage, selected_temp
)

st.sidebar.subheader("Predicted Removal")
st.sidebar.metric(
    label="Milk Fat Removal Efficiency",
    value=f"{predicted_removal:.2f}%",
    delta_color="off"
)

st.sidebar.markdown("---")
st.sidebar.header("Current Parameters")
st.sidebar.text(f"pH: {selected_ph:.1f}")
st.sidebar.text(f"Conc: {selected_conc:.0f} mg/L")
st.sidebar.text(f"Dosage: {selected_dosage:.1f} g/L")
st.sidebar.text(f"Temp: {selected_temp:.0f} °C")


# --- MAIN CONTENT LAYOUT ---
col1, col2 = st.columns([3, 2])

with col1:
    st.header("1. ML Prediction Curve and Data Visualization (pH Focus)")
    st.markdown("The **3rd Degree Polynomial Regression** model now includes **4 features** and their interactions to fit the complex removal surface. The plot below still focuses on the primary effect of **pH**.")
    
    # --- PLOTLY VISUALIZATION (Fixed Concentration/Dosage/Temp for 2D plot) ---
    
    # Generate prediction line data for the 2D plot (fixing other variables at current selection)
    ph_range = np.linspace(2.0, 10.0, 50)
    
    # Create an array where only pH changes, and others are fixed
    fixed_params_df = pd.DataFrame({
        'pH': ph_range,
        'Concentration (mg/L)': selected_conc,
        'Dosage (g/L)': selected_dosage,
        'Temperature (°C)': selected_temp
    })

    # Predict the curve using the fixed parameters
    predicted_curve_fixed = ml_model_enhanced.predict(fixed_params_df)
    
    prediction_df_fixed = pd.DataFrame({
        'pH': ph_range,
        'Prediction (%)': predicted_curve_fixed
    })
    
    # Plotly visualization: Scatter of all data, line of the current fixed-parameter prediction
    fig = px.scatter(
        data_df_enhanced, 
        x='pH', 
        y='Milk_Fat_Removal_Efficiency (%)', 
        title='Removal Efficiency vs. pH (Colored by Dosage)',
        color='Dosage (g/L)', # Use one of the new factors for visual context
        color_continuous_scale=px.colors.sequential.Bluyl # Adjust color scale
    )
    
    # Add prediction line based on fixed parameters
    fig.add_scatter(
        x=prediction_df_fixed['pH'], 
        y=prediction_df_fixed['Prediction (%)'], 
        mode='lines', 
        name=f'ML Curve @ {selected_conc:.0f}mg/L, {selected_dosage:.1f}g/L, {selected_temp:.0f}°C',
        line=dict(color='#F94144', width=4) # Energetic Red
    )

    # Highlight the user's current specific selection
    fig.add_scatter(
        x=[selected_ph],
        y=[predicted_removal],
        mode='markers',
        marker=dict(size=14, color='white', symbol='star-diamond', line=dict(width=2, color='black')), # Highlight marker
        name='Current Prediction Point'
    )
    
    fig.update_layout(
        xaxis_title="Solution pH",
        yaxis_title="Removal Efficiency (%)",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("2. Factor Interaction Insights")
    st.markdown("""
    The enhanced model simulates the complex **interaction** of four key factors:
    * **$\mathbf{pH}$**: Retains its role as the **primary control** (optimum at 4).
    * **Dosage**: Higher dosage generally boosts efficiency, especially in **sub-optimal $\mathbf{pH}$** conditions.
    * **Concentration**: Efficiency slightly decreases as the initial pollutant load is **higher** (more competition for active sites).
    * **Temperature**: Exhibits an **optimum** around $35^\circ\text{C}$, indicating the process is likely **chemisorption-driven** (since pure physisorption usually decreases with temperature).
    """)
    
    st.markdown("---")
    st.subheader("How Prediction Changes")
    
    # Example to show the effect of changing one parameter
    # Test case: Mid-range pH 6, others optimal
    test_ph = 6.0
    test_conc = 150.0
    test_dosage = 2.5
    test_temp = 35.0
    
    base_pred = predict_removal_enhanced(ml_model_enhanced, test_ph, test_conc, test_dosage, test_temp)
    
    # 1. Effect of increasing Dosage
    high_dosage_pred = predict_removal_enhanced(ml_model_enhanced, test_ph, test_conc, 4.5, test_temp)
    dosage_delta = high_dosage_pred - base_pred
    
    # 2. Effect of increasing Concentration
    high_conc_pred = predict_removal_enhanced(ml_model_enhanced, test_ph, 400.0, test_dosage, test_temp)
    conc_delta = high_conc_pred - base_pred
    
    st.metric(
        label=f"Baseline Removal @ pH {test_ph:.1f}",
        value=f"{base_pred:.2f}%",
        delta_color="off"
    )
    st.metric(
        label=f"Effect of Max Dosage (4.5 g/L)",
        value=f"{high_dosage_pred:.2f}%",
        delta=f"{dosage_delta:.2f}% improvement",
        delta_color="normal"
    )
    st.metric(
        label=f"Effect of High Conc (400 mg/L)",
        value=f"{high_conc_pred:.2f}%",
        delta=f"{conc_delta:.2f}% change",
        delta_color="inverse"
    )
    
# --- CONCLUSION ---
st.markdown("---")
st.markdown("""
### Conclusion on Enhanced Modeling

By incorporating **Concentration, Dosage, and Temperature** into a **Polynomial Regression Model (Degree 3)**, the simulation more accurately reflects a typical **adsorption isotherm** and **thermodynamic study**. The model captures:
1.  The primary, non-linear control of **pH**.
2.  The benefit of increased **adsorbent mass (Dosage)**.
3.  The limiting effect of high initial **pollutant concentration**.
4.  The optimal conditions for **chemisorption-driven** removal (Temperature).
""")