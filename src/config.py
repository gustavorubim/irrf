import pandas as pd
from datetime import datetime

# --- Constants ---
# Use a recent year for historical data download
CURRENT_YEAR = datetime.now().year
# Fetch last full year data
DATA_URL = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={CURRENT_YEAR-1}"

# Maturities provided by the Treasury data (update if source changes)
# Keys: Column names in Treasury XML, Values: Maturity in years
MATURITY_MAP = {
    'BC_1MONTH': 1/12,
    'BC_2MONTH': 2/12,
    'BC_3MONTH': 3/12,
    'BC_4MONTH': 4/12, # Often missing, handle carefully
    'BC_6MONTH': 6/12,
    'BC_1YEAR': 1.0,
    'BC_2YEAR': 2.0,
    'BC_3YEAR': 3.0,
    'BC_5YEAR': 5.0,
    'BC_7YEAR': 7.0,
    'BC_10YEAR': 10.0,
    'BC_20YEAR': 20.0,
    'BC_30YEAR': 30.0
}
MATURITIES_LIST = list(MATURITY_MAP.values())

# Simulation Parameters
SIMULATION_YEARS = 3
DAYS_PER_YEAR = 252 # Approximate trading days
SIMULATION_STEPS = SIMULATION_YEARS * DAYS_PER_YEAR
MC_RUNS = 1000 # Number of Monte Carlo simulations

# File Paths
DATA_DIR = "data"
PLOTS_DIR = "plots"
RAW_YIELDS_PATH = f"{DATA_DIR}/raw_yields.parquet"
NS_PARAMS_PATH = f"{DATA_DIR}/ns_parameters.parquet"
VAR_RESULTS_PATH = f"{DATA_DIR}/var_parameters.pickle" # Using pickle for VARResultsWrapper
SINGLE_SIM_CURVES_PATH = f"{DATA_DIR}/single_simulation_curves.parquet"

MC_DATA_DIR = f"{DATA_DIR}/monte_carlo"
MC_SIM_PATHS_PATH = f"{MC_DATA_DIR}/simulation_paths.parquet"
MC_ANALYSIS_RESULTS_PATH = f"{MC_DATA_DIR}/analysis_results.parquet"

# Plot Paths
SIM_SURFACE_PLOT_PATH = f"{PLOTS_DIR}/simulation_surface.png"
BETA_HIST_PLOT_PATH = f"{PLOTS_DIR}/beta_histograms.png"
DRAWDOWN_PLOT_PATH = f"{PLOTS_DIR}/drawdown_analysis.png"

# Portfolio Definition (Example: 50% 2Y ZC, 50% 10Y ZC)
PORTFOLIO_DEF = {
    'weights': [0.5, 0.5],
    'maturities': [2.0, 10.0]
}