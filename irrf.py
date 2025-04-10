import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from io import StringIO
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output (optional)
warnings.filterwarnings("ignore")

# --- Constants ---
# Use a recent year for historical data download
CURRENT_YEAR = datetime.now().year
DATA_URL = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={CURRENT_YEAR-1}" # Fetch last full year data
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
SIMULATION_YEARS = 3
DAYS_PER_YEAR = 252 # Approximate trading days
SIMULATION_STEPS = SIMULATION_YEARS * DAYS_PER_YEAR

# --- 1. Download US Yield Curve Data ---

def download_treasury_yield_data(url):
    """Downloads and parses Treasury yield curve XML data."""
    print(f"Downloading data from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        print("Download successful.")
        
        # Parse XML
        xml_content = response.content
        # Find the namespace - it can change, so find it dynamically
        root = ET.fromstring(xml_content)
        # Namespace is usually defined like 'm' or 'd', find it in the root tag attributes
        namespace = {'ns': root.tag.split('}')[0].strip('{')} # Extract namespace URI

        data = []
        # Find all 'entry' elements, then 'content', then 'properties'
        for entry in root.findall('.//ns:entry', namespace):
            properties = entry.find('.//ns:content/m:properties', {'m': 'http://schemas.microsoft.com/ado/2007/08/dataservices/metadata', 'ns': namespace['ns']})
            if properties is None: continue

            record = {}
            # Extract date - it's usually under a 'd:NEW_DATE' tag
            date_tag = properties.find('.//d:NEW_DATE', {'d': 'http://schemas.microsoft.com/ado/2007/08/dataservices'})
            if date_tag is not None and date_tag.text:
                 # Date might have time component, strip it
                 record['Date'] = pd.to_datetime(date_tag.text.split('T')[0])
            else:
                 continue # Skip if date is missing

            # Extract yields
            valid_record = True
            for key in MATURITY_MAP.keys():
                 yield_tag = properties.find(f'.//d:{key}', {'d': 'http://schemas.microsoft.com/ado/2007/08/dataservices'})
                 if yield_tag is not None and yield_tag.text:
                     try:
                         record[key] = float(yield_tag.text)
                     except ValueError:
                         record[key] = np.nan # Handle non-numeric entries
                         valid_record = False
                 else:
                     record[key] = np.nan # Handle missing yield data points
                     # Optional: Decide if a record is invalid if certain key maturities are missing
                     # if key in ['BC_3MONTH', 'BC_2YEAR', 'BC_10YEAR']:
                     #     valid_record = False

            if valid_record and record: # Only add if date was found
                data.append(record)

        if not data:
            print("Warning: No data parsed. Check XML structure or URL.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.set_index('Date')
        df = df.sort_index()
        # Convert yields to percentage (they are usually published as %)
        # df = df / 100.0 # Uncomment if data is not already in decimal form

        # Use only the maturities defined in MATURITY_MAP and rename columns to years
        df = df[list(MATURITY_MAP.keys())]
        df.columns = [MATURITY_MAP[col] for col in df.columns]
        
        # Drop rows/cols with too many NaNs if necessary
        df = df.dropna(axis=0, how='any') # Drop rows with ANY NaNs for simplicity
        print(f"Data processed into DataFrame. Shape: {df.shape}")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        # Optionally save problematic content for debugging
        # with open("error_yield_curve.xml", "wb") as f:
        #     f.write(xml_content)
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during download/parsing: {e}")
        return pd.DataFrame()


# --- 2. Fit Nelson-Siegel Model ---

def nelson_siegel_yields(params, maturities):
    """Calculates yields using the Nelson-Siegel formula."""
    beta0, beta1, beta2, lambda_ = params
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-lambda_ * maturities)) / (lambda_ * maturities)
    term3 = beta2 * ((1 - np.exp(-lambda_ * maturities)) / (lambda_ * maturities) - np.exp(-lambda_ * maturities))
    
    # Handle potential division by zero for maturity=0 if lambda*maturity is tiny
    # For maturity -> 0, the limit of term2 is beta1, term3 is 0.
    # However, maturities array typically doesn't include 0.
    # If maturities contains 0, special handling is needed. For standard NS, lambda_ > 0.
    # A small epsilon can prevent NaN if maturities are extremely small but non-zero.
    epsilon = 1e-8
    term2 = np.where(np.abs(lambda_ * maturities) < epsilon, beta1, term2)
    term3 = np.where(np.abs(lambda_ * maturities) < epsilon, 0, term3)

    return term1 + term2 + term3

def error_function(params, maturities, actual_yields):
    """Calculates the sum of squared errors for Nelson-Siegel fit."""
    # Penalize non-positive lambda heavily
    if params[3] <= 0:
        return 1e10 # Return a large number
    
    model_yields = nelson_siegel_yields(params, maturities)
    error = np.sum((model_yields - actual_yields) ** 2)
    return error

def fit_nelson_siegel_on_date(maturities, yields_on_date):
    """Fits the NS model for a single date using optimization."""
    # Initial guesses for parameters [beta0, beta1, beta2, lambda_]
    # Reasonable guesses: beta0=long end yield, beta1=slope (short-long), beta2=curvature, lambda=around 1-2
    initial_lambda = 1.5 
    initial_beta0 = yields_on_date[-1] # Use longest yield as initial guess for level
    initial_beta1 = yields_on_date[0] - yields_on_date[-1] # Use short-long spread for slope
    # Find mid-term maturity index for curvature guess
    mid_maturity_idx = np.argmin(np.abs(maturities - 2.0)) # e.g., around 2 years
    initial_beta2 = yields_on_date[mid_maturity_idx] - (initial_beta0 + initial_beta1 * (1 - np.exp(-initial_lambda * maturities[mid_maturity_idx])) / (initial_lambda * maturities[mid_maturity_idx]))
    
    p0 = [initial_beta0, initial_beta1, initial_beta2, initial_lambda] 
    
    # Bounds for parameters (optional but recommended, especially for lambda > 0)
    bounds = [(None, None), (None, None), (None, None), (1e-6, None)] # lambda > 0
    
    result = minimize(error_function, p0, args=(maturities, yields_on_date),
                      method='L-BFGS-B', # Method that handles bounds
                      bounds=bounds)
    
    if result.success:
        return result.x # Return the fitted parameters [beta0, beta1, beta2, lambda_]
    else:
        print(f"Warning: Optimization failed for a date. Message: {result.message}")
        return np.full(4, np.nan) # Return NaNs if optimization fails

def fit_historical_ns(yield_data):
    """Fits the NS model to each date in the historical data."""
    print("Fitting Nelson-Siegel model to historical data...")
    maturities = yield_data.columns.to_numpy(dtype=float)
    fitted_params = []
    dates = []

    for date, row in yield_data.iterrows():
        yields_on_date = row.to_numpy(dtype=float)
        # Ensure no NaNs in this specific row before fitting
        if not np.isnan(yields_on_date).any():
            params = fit_nelson_siegel_on_date(maturities, yields_on_date)
            fitted_params.append(params)
            dates.append(date)
        else:
            print(f"Skipping date {date} due to NaN values.")

    params_df = pd.DataFrame(fitted_params, columns=['beta0', 'beta1', 'beta2', 'lambda'], index=dates)
    # Drop any rows where fitting failed
    params_df = params_df.dropna()
    print(f"Nelson-Siegel fitting complete. Estimated factors shape: {params_df.shape}")
    return params_df

# --- 3. Simulate Factors using VAR and Generate Yield Curves ---

def simulate_factors_var(factors_df, n_steps):
    """Fits a VAR model to factors and simulates future paths."""
    print("Fitting VAR model to estimated factors...")
    # Use only beta0, beta1, beta2 for VAR (lambda often assumed constant or less dynamic)
    factors_to_model = factors_df[['beta0', 'beta1', 'beta2']]
    
    # Simple VAR(1) model
    var_model = VAR(factors_to_model)
    try:
        # Fit VAR model, choosing lag order (e.g., 1)
        var_results = var_model.fit(maxlags=1, ic=None) # Use lag p=1 for simplicity
        print(f"VAR(1) model fitted. Lag order: {var_results.k_ar}")

        # Get the last observed values to start the forecast
        last_observation = factors_to_model.values[-var_results.k_ar:]
        
        print(f"Simulating factors for {n_steps} steps...")
        # Forecast n_steps ahead
        simulated_factor_values = var_results.forecast(y=last_observation, steps=n_steps)
        
        # Combine with the lambda parameter (use the last observed value)
        last_lambda = factors_df['lambda'].iloc[-1]
        simulated_lambda = np.full((n_steps, 1), last_lambda)
        
        # Combine beta factors and lambda
        simulated_params = np.hstack((simulated_factor_values, simulated_lambda))
        
        # Create a DataFrame for simulated parameters
        last_date = factors_df.index[-1]
        sim_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_steps, freq='B') # Business day frequency
        sim_params_df = pd.DataFrame(simulated_params, columns=['beta0', 'beta1', 'beta2', 'lambda'], index=sim_dates)
        
        print("Factor simulation complete.")
        return sim_params_df

    except Exception as e:
        print(f"Error fitting or simulating VAR model: {e}")
        return pd.DataFrame()

def generate_simulated_curves(simulated_params_df, maturities):
    """Generates yield curves from simulated NS parameters."""
    print("Generating simulated yield curves...")
    simulated_curves = []
    for index, params in simulated_params_df.iterrows():
        yield_curve = nelson_siegel_yields(params.values, maturities)
        simulated_curves.append(yield_curve)
    
    curves_df = pd.DataFrame(simulated_curves, index=simulated_params_df.index, columns=maturities)
    print("Simulated yield curve generation complete.")
    return curves_df

# --- 4. Plot Simulation Results ---

def plot_simulation_surface(simulated_curves_df):
    """Plots the simulated yield curve evolution as a 3D surface."""
    print("Plotting simulation results...")
    if simulated_curves_df.empty:
        print("No simulated data to plot.")
        return
        
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data for surface plot
    X = simulated_curves_df.columns.values # Maturities
    Y = np.arange(len(simulated_curves_df.index)) # Time steps (as numbers)
    X, Y = np.meshgrid(X, Y)
    Z = simulated_curves_df.values # Yields

    # Create the surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    # Customize the plot
    ax.set_title(f'Simulated US Treasury Yield Curve Evolution ({SIMULATION_YEARS} Years)')
    ax.set_xlabel('Maturity (Years)')
    ax.set_ylabel('Simulation Time Steps (Days)')
    ax.set_zlabel('Yield (%)') # Assuming yields are in %

    # Customize Y ticks to show dates (optional, can get crowded)
    step = max(1, len(simulated_curves_df.index) // 5) # Show approx 5 date labels
    tick_indices = np.arange(0, len(simulated_curves_df.index), step)
    tick_labels = [simulated_curves_df.index[i].strftime('%Y-%m-%d') for i in tick_indices]
    ax.set_yticks(tick_indices)
    ax.set_yticklabels(tick_labels, rotation=-15, ha='left')


    # Add a color bar
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Yield (%)')

    # Improve layout and display
    plt.tight_layout()
    plt.show()
    print("Plotting complete.")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Download Data
    historical_yields = download_treasury_yield_data(DATA_URL)

    if not historical_yields.empty:
        # Extract maturities vector from columns
        maturities_vector = historical_yields.columns.to_numpy(dtype=float)

        # 2. Fit Nelson-Siegel Historically
        historical_ns_factors = fit_historical_ns(historical_yields)

        if not historical_ns_factors.empty:
             # Check if enough data for VAR
             if len(historical_ns_factors) < 10: # Need some data points for VAR
                  print("Error: Not enough historical factor data points to fit VAR model.")
             else:
                  # 3. Simulate Factors and Generate Curves
                  simulated_factors = simulate_factors_var(historical_ns_factors, SIMULATION_STEPS)

                  if not simulated_factors.empty:
                       simulated_yield_curves = generate_simulated_curves(simulated_factors, maturities_vector)

                       # 4. Plot Simulation
                       plot_simulation_surface(simulated_yield_curves)
                  else:
                       print("Factor simulation failed. Cannot generate or plot curves.")
        else:
            print("Nelson-Siegel fitting failed. Cannot proceed.")
    else:
        print("Data download failed. Cannot proceed.")