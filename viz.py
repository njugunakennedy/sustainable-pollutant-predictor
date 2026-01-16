import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize

# --- CONFIGURATION ---
CSV_FILE = 'biochar_milk_fat_removal.csv'  # Your dataset
TARGET_COL = 'Removal_Efficiency'
FEATURES = ['pH', 'Temperature', 'Concentration', 'Adsorbent_Dosage', 'Volume']

# User Defined Bounds for Optimization
BOUNDS = [
    (2.0, 9.9),     # pH
    (20.0, 50.0),   # Temperature
    (55.0, 500.0),  # Concentration
    (0.11, 2.99),   # Dosage
    (250.0, 1000.0) # Volume
]

# 1. LOAD DATA
try:
    df = pd.read_csv(CSV_FILE)
    X = df[FEATURES]
    y = df[TARGET_COL]
    print(f"Loaded {len(df)} rows.")
except FileNotFoundError:
    print("Error: CSV file not found. Please run the data generator first.")
    exit()

# 2. BUILD SECOND-ORDER POLYNOMIAL MODEL
# Degree=2 generates: A, B, A^2, B^2, A*B
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# 3. MODEL VALIDATION METRICS
y_pred = model.predict(X_poly)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("\n--- Model Performance ---")
print(f"R² Score: {r2:.4f}")
print(f"RMSE:     {rmse:.4f} %")

# 4. EXPLICIT MODEL EQUATION
# We extract coefficients to print the formula Y = Intercept + C1*X1 ...
feature_names = poly.get_feature_names_out(FEATURES)
coefficients = model.coef_
intercept = model.intercept_

equation = f"Efficiency = {intercept:.4f}"
for name, coef in zip(feature_names, coefficients):
    # Skip the bias term (1) if present in names, though coef usually handles it
    if name == "1" or abs(coef) < 0.001: continue 
    # Replace spaces with multiplication for readability
    clean_name = name.replace(" ", "*")
    equation += f" + ({coef:.4f} * {clean_name})"

print("\n--- Explicit Model Equation ---")
print(equation)

# 5. OPTIMIZATION (MAXIMIZE EFFICIENCY)
# Scipy minimizes by default, so we minimize the negative efficiency
def objective_function(x):
    # x is an array [pH, Temp, Conc, Dosage, Volume]
    # We must reshape it to (1, -1) and transform it to poly features
    x_transformed = poly.transform([x])
    predicted_efficiency = model.predict(x_transformed)
    return -predicted_efficiency[0]  # Negative for maximization

# Initial guess (start at the midpoint of bounds)
x0 = [5.5, 35, 275, 1.5, 600]

# Run Optimization
result = minimize(objective_function, x0, bounds=BOUNDS, method='L-BFGS-B')

optimal_inputs = result.x
max_efficiency = -result.fun

print("\n--- Optimization Results ---")
print(f"Maximum Predicted Efficiency: {max_efficiency:.2f}%")
print("Optimal Operating Conditions:")
for i, feature in enumerate(FEATURES):
    print(f"  {feature}: {optimal_inputs[i]:.2f}")

# 6. VISUALIZATIONS

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

# A. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df[FEATURES + [TARGET_COL]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('viz_output/correlation_heatmap.png')
plt.close()

# B. Actual vs Predicted & Residuals
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs Predicted
axes[0].scatter(y, y_pred, color='#3b82f6', alpha=0.6, edgecolor='k')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Efficiency')
axes[0].set_ylabel('Predicted Efficiency')
axes[0].set_title(f'Actual vs Predicted (R²={r2:.3f})')

# Residuals vs Predicted
residuals = y - y_pred
axes[1].scatter(y_pred, residuals, color='#ef4444', alpha=0.6, edgecolor='k')
axes[1].axhline(0, color='black', linestyle='--')
axes[1].set_xlabel('Predicted Efficiency')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot (Homoscedasticity Check)')
plt.tight_layout()
plt.savefig('viz_output/actual_vs_predicted.png')
plt.close()

# C. 3D Response Surface & 2D Contour
# We visualize pH vs Dosage (holding others constant at optimal values)
idx_x = 0 # pH
idx_y = 3 # Dosage
fixed_vals = optimal_inputs.copy() # Hold Temp, Conc, Vol constant

# Create grid
x_range = np.linspace(BOUNDS[idx_x][0], BOUNDS[idx_x][1], 50) # pH range
y_range = np.linspace(BOUNDS[idx_y][0], BOUNDS[idx_y][1], 50) # Dosage range
X_grid, Y_grid = np.meshgrid(x_range, y_range)
Z_grid = np.zeros_like(X_grid)

# Fill grid with predictions
for i in range(X_grid.shape[0]):
    for j in range(X_grid.shape[1]):
        # Construct the full input vector
        sample = fixed_vals.copy()
        sample[idx_x] = X_grid[i, j]
        sample[idx_y] = Y_grid[i, j]
        # Transform and predict
        sample_poly = poly.transform([sample])
        Z_grid[i, j] = model.predict(sample_poly)[0]

# Plot 3D
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none', alpha=0.9)
ax1.set_xlabel('pH')
ax1.set_ylabel('Dosage (g/L)')
ax1.set_zlabel('Efficiency (%)')
ax1.set_title('3D Response Surface: pH vs Dosage')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

# Plot 2D Contour
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X_grid, Y_grid, Z_grid, levels=20, cmap='viridis')
ax2.set_xlabel('pH')
ax2.set_ylabel('Dosage (g/L)')
ax2.set_title('2D Contour Plot')
plt.colorbar(contour, ax=ax2, label='Efficiency (%)')
plt.tight_layout()
plt.savefig('viz_output/response_surface.png')
plt.close()

# D. Sensitivity Analysis (Perturbation Plot)
# Vary each factor from -50% to +50% of its range (normalized) around the optimum
plt.figure(figsize=(10, 6))

# Normalize range for plotting (-1 to 1)
for i, name in enumerate(FEATURES):
    # Create 50 points across the variable's full bounds
    var_range = np.linspace(BOUNDS[i][0], BOUNDS[i][1], 50)
    efficiencies = []
    
    for val in var_range:
        temp_input = optimal_inputs.copy()
        temp_input[i] = val
        pred = model.predict(poly.transform([temp_input]))[0]
        efficiencies.append(pred)
    
    # Plot line
    # We plot against the actual value, but you can also plot against deviation %
    plt.plot(var_range, efficiencies, label=name, linewidth=2)

plt.axvline(optimal_inputs[0], color='gray', linestyle=':', alpha=0.5) # Mark Optimal pH as ref
plt.title('Sensitivity Analysis (One-Factor-at-a-Time)')
plt.xlabel('Factor Value (Varying across its range)')
plt.ylabel('Predicted Efficiency (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('viz_output/sensitivity_analysis.png')
plt.close()