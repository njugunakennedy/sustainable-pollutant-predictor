import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Biochar Adsorption Optimizer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. DATA GENERATION & LOADING ---
@st.cache_data
def load_or_generate_data():
    """
    Generates the synthetic dataset based on the physics-informed logic 
    derived in our previous steps (Project: Milk Fat Removal).
    """
    np.random.seed(42)
    NUM_SAMPLES = 300
    
    data = {
        'pH': np.round(np.random.uniform(2, 9.9, NUM_SAMPLES), 1),
        'Temperature': np.random.choice([20, 25, 30, 35, 40, 45, 50], NUM_SAMPLES),
        'Concentration': np.round(np.random.uniform(55, 500, NUM_SAMPLES), 0),
        'Adsorbent_Dosage': np.round(np.random.uniform(0.11, 2.99, NUM_SAMPLES), 2),
        'Volume': np.random.choice([250, 500, 1000], NUM_SAMPLES)
    }
    df = pd.DataFrame(data)

    # Simulation Logic (The "Ground Truth" we reverse-engineered)
    def calculate_efficiency(row):
        eff = 50.0 
        eff += 25 * np.log(row['Adsorbent_Dosage'] + 1) # Dosage log growth
        eff -= 1.8 * ((row['pH'] - 6.2) ** 2)           # pH quadratic penalty (Optimum ~6.2)
        eff -= 0.15 * ((row['Temperature'] - 30) ** 2)  # Temp quadratic penalty (Optimum ~30)
        eff -= 0.04 * (row['Concentration'] - 100)      # Concentration penalty
        if row['Volume'] > 500: eff -= 2.0              # Volume penalty
        eff += 0.02 * (row['Adsorbent_Dosage'] * row['Concentration']) # Interaction
        eff += np.random.normal(0, 3.0)                 # Noise
        return eff

    df['Removal_Efficiency'] = df.apply(calculate_efficiency, axis=1)
    df['Removal_Efficiency'] = df['Removal_Efficiency'].clip(0, 99.9).round(2)
    return df

# --- 2. MODEL TRAINING ---
@st.cache_resource
def train_model(df):
    """Trains the 2nd Order Polynomial Regression Model."""
    features = ['pH', 'Temperature', 'Concentration', 'Adsorbent_Dosage', 'Volume']
    X = df[features]
    y = df['Removal_Efficiency']
    
    # Polynomial Features (Degree 2 for Interactions)
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Validation Metrics
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    return model, poly, features, r2, rmse

# --- LOAD DATA & MODEL ---
df = load_or_generate_data()
model, poly, feature_names, r2_val, rmse_val = train_model(df)

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("🎛️ Process Parameters")
st.sidebar.markdown("Adjust inputs to predict removal efficiency.")

input_ph = st.sidebar.slider("pH Level", 2.0, 9.9, 6.2, 0.1)
input_temp = st.sidebar.slider("Temperature (°C)", 20, 50, 30, 5)
input_conc = st.sidebar.slider("Initial Conc. (mg/L)", 55, 500, 200, 5)
input_dose = st.sidebar.slider("Adsorbent Dosage (g/L)", 0.11, 2.99, 1.5, 0.1)
input_vol = st.sidebar.select_slider("Volume (mL)", options=[250, 500, 1000], value=500)

# Real-time Prediction
input_data = np.array([[input_ph, input_temp, input_conc, input_dose, input_vol]])
input_poly = poly.transform(input_data)
prediction = model.predict(input_poly)[0]
prediction = np.clip(prediction, 0, 100) # Clamp between 0-100

st.sidebar.markdown("---")
st.sidebar.subheader("Current Prediction")
st.sidebar.metric(
    label="Removal Efficiency", 
    value=f"{prediction:.2f}%", 
    delta=f"R² of Model: {r2_val:.3f}"
)

# --- MAIN PAGE ---
st.title("🧪 Biochar Adsorption Optimization")
st.markdown(f"""
**Project:** Pollutant Removal from Milk Processing Wastewater using Biochar.  
**Model:** Response Surface Methodology (RSM) via 2nd-Order Polynomial Regression.  
**Accuracy:** The model explains **{r2_val*100:.1f}%** of the variance in the data ($RMSE = {rmse_val:.2f}\%$).
""")

# --- TABS FOR ORGANIZED VIEW ---
tab1, tab2, tab3 = st.tabs(["📈 Optimization & Visuals", "📊 Model Diagnostics", "📝 Equation"])

# TAB 1: OPTIMIZATION & 3D PLOTS
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🚀 Numerical Optimization")
        st.markdown("Find the theoretical maximum efficiency using **SciPy L-BFGS-B**.")
        
        if st.button("Run Optimizer"):
            # Optimization Function
            def objective(x):
                # x = [pH, Temp, Conc, Dosage, Vol]
                return -model.predict(poly.transform([x.reshape(1, -1)]))[0]

            bounds = [(2.0, 9.9), (20, 50), (55, 500), (0.11, 2.99), (250, 1000)]
            x0 = [6.0, 30.0, 200.0, 1.5, 500.0] # Initial guess
            
            res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            
            st.success(f"**Max Efficiency Found: {-res.fun:.2f}%**")
            st.markdown(f"""
            **Optimal Conditions:**
            * **pH:** {res.x[0]:.2f}
            * **Temp:** {res.x[1]:.1f} °C
            * **Conc:** {res.x[2]:.0f} mg/L
            * **Dosage:** {res.x[3]:.2f} g/L
            * **Volume:** {res.x[4]:.0f} mL
            """)
        else:
            st.info("Click to calculate optimal parameters.")

    with col2:
        st.subheader("Interactive 3D Response Surface")
        st.markdown(f"Visualizing **pH vs. Dosage** (holding Temp={input_temp}°C, Conc={input_conc}mg/L constant).")

        # Generate Grid for 3D Plot
        x_range = np.linspace(2.0, 9.9, 30) # pH
        y_range = np.linspace(0.11, 2.99, 30) # Dosage
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        Z_grid = np.zeros_like(X_grid)

        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                # Create a sample vector using grid values + user sidebar inputs for others
                sample = np.array([[X_grid[i,j], input_temp, input_conc, Y_grid[i,j], input_vol]])
                Z_grid[i,j] = model.predict(poly.transform(sample))[0]

        # Plotly 3D Surface 
        fig = go.Figure(data=[go.Surface(z=Z_grid, x=X_grid, y=Y_grid, colorscale='Viridis')])
        fig.update_layout(
            title='Response Surface: pH vs Dosage', 
            scene=dict(
                xaxis_title='pH',
                yaxis_title='Dosage (g/L)',
                zaxis_title='Efficiency (%)'
            ),
            height=500,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: DIAGNOSTICS
with tab2:
    st.subheader("Model Validation")
    col_d1, col_d2 = st.columns(2)
    
    # 1. Actual vs Predicted
    with col_d1:
        y_pred_all = model.predict(poly.transform(df[feature_names]))
        fig_avp = px.scatter(
            x=df['Removal_Efficiency'], y=y_pred_all, 
            labels={'x': 'Actual Efficiency (%)', 'y': 'Predicted Efficiency (%)'},
            title="Actual vs. Predicted",
            opacity=0.6
        )
        fig_avp.add_shape(type="line", line=dict(dash="dash", color="red"),
            x0=0, x1=100, y0=0, y1=100)
        st.plotly_chart(fig_avp, use_container_width=True)
        
    # 2. Residuals
    with col_d2:
        residuals = df['Removal_Efficiency'] - y_pred_all
        fig_res = px.scatter(
            x=y_pred_all, y=residuals,
            labels={'x': 'Predicted Efficiency (%)', 'y': 'Residuals'},
            title="Residual Plot",
            opacity=0.6, color_discrete_sequence=['#ef4444']
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_res, use_container_width=True)

# TAB 3: EQUATION
with tab3:
    st.subheader("Explicit Model Equation")
    st.markdown("This equation represents the model.")
    
    # Extract coefficients
    coefs = model.coef_
    intercept = model.intercept_
    feature_names_poly = poly.get_feature_names_out(feature_names)
    
    equation_str = f"$$ Y = {intercept:.2f} "
    
    for name, coef in zip(feature_names_poly, coefs):
        if abs(coef) > 0.01 and name != "1": # Filter small coeffs
            sign = "+" if coef > 0 else "-"
            # Format clean names (e.g., pH^2)
            clean_name = name.replace(" ", " \\cdot ")
            equation_str += f"{sign} {abs(coef):.3f}({clean_name}) "
    
    equation_str += "$$"
    st.latex(equation_str)