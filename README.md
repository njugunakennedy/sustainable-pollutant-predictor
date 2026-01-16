# Sustainable Pollutant Predictor

## Project Overview
This project focuses on optimizing the removal of pollutants (specifically milk fat) from wastewater using biochar adsorption. It utilizes a machine learning approach (2nd-order Polynomial Regression) to model the relationship between process parameters and removal efficiency.

The project includes:
1.  **Synthetic Data Generation**: Simulates experimental data based on physics-informed logic.
2.  **Machine Learning Model**: Trains a regression model to predict removal efficiency.
3.  **Interactive Web App**: A Streamlit dashboard for real-time prediction, visualization, and optimization.
4.  **Analysis Scripts**: Standalone scripts for generating static visualizations and analysis.

## Features
-   **Predictive Modeling**: Estimates removal efficiency based on pH, Temperature, Concentration, Adsorbent Dosage, and Volume.
-   **Optimization**: Uses SciPy's L-BFGS-B algorithm to find the optimal operating conditions for maximum efficiency.
-   **Visualizations**:
    -   3D Response Surface Plots (pH vs. Dosage).
    -   Actual vs. Predicted scatter plots.
    -   Residual plots for model diagnostics.
    -   Correlation heatmaps.

## Files
-   `stream.py`: The main Streamlit application.
-   `viz.py`: A script to generate and save static analysis plots to the `viz_output/` directory.
-   `biochar_milk_fat_removal.csv`: The dataset used for training and analysis.
-   `requirements.txt`: List of Python dependencies.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/njugunakennedy/sustainable-pollutant-predictor.git
    cd sustainable-pollutant-predictor
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Web App
To launch the interactive dashboard:
```bash
streamlit run stream.py
```
This will open the application in your default web browser.

### Generating Analysis Plots
To generate static plots (saved to `viz_output/`):
```bash
python viz.py
```

## Technologies Used
-   **Python**: Core programming language.
-   **Streamlit**: Web application framework.
-   **Scikit-Learn**: Machine learning (Polynomial Features, Linear Regression).
-   **SciPy**: Numerical optimization.
-   **Plotly**: Interactive visualizations.
-   **Matplotlib & Seaborn**: Static plotting.
-   **Pandas & NumPy**: Data manipulation and numerical operations.
