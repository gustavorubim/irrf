import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import pickle
import warnings
from datetime import timedelta

# Import configuration constants and NS functions
from . import config
from .nelson_siegel import nelson_siegel_yields
from .data_loader import save_data_to_parquet, load_data_from_parquet # For saving curves

# Suppress warnings for cleaner output (optional)
warnings.filterwarnings("ignore")

# --- VAR Model Fitting ---

def fit_var_model(factors_df, maxlags=1, factors_to_model=['beta0', 'beta1', 'beta2']):
    """
    Fits a VAR model to the specified historical factor data.

    Args:
        factors_df (pd.DataFrame): DataFrame of historical NS factors, indexed by date.
        maxlags (int): The maximum number of lags to check for the VAR model.
                       Use None to let the information criterion choose. Use 1 for VAR(1).
        factors_to_model (list): List of column names in factors_df to include in the VAR.

    Returns:
        statsmodels.tsa.vector_ar.var_model.VARResultsWrapper: The fitted VAR model results object,
                                                               or None if fitting fails.
    """
    print(f"Fitting VAR({maxlags if maxlags else 'auto'}) model to factors: {factors_to_model}...")
    if factors_df.empty or len(factors_df) < maxlags + 5: # Need sufficient data points
        print(f"Error: Not enough historical factor data points ({len(factors_df)}) to fit VAR model reliably.")
        return None

    factors_subset = factors_df[factors_to_model]

    try:
        var_model = VAR(factors_subset)
        # Fit VAR model, choosing lag order (e.g., 1 or use AIC/BIC)
        # Using ic=None forces the use of maxlags
        var_results = var_model.fit(maxlags=maxlags, ic=None)
        print(f"VAR model fitted. Lag order: {var_results.k_ar}")
        print("Fitted VAR parameters (coefficients):\n", var_results.params)
        print("\nResidual Covariance Matrix (Sigma_u):\n", var_results.sigma_u)
        return var_results

    except Exception as e:
        print(f"Error fitting VAR model: {e}")
        # import traceback
        # print(traceback.format_exc())
        return None

# --- Factor Simulation ---

def forecast_factors_deterministic(var_results, factors_df, n_steps=config.SIMULATION_STEPS, factors_modeled=['beta0', 'beta1', 'beta2']):
    """
    Generates a single deterministic forecast path for the factors using the fitted VAR model.

    Args:
        var_results (VARResultsWrapper): The fitted VAR model results.
        factors_df (pd.DataFrame): The historical factors DataFrame (used for initial values and lambda).
        n_steps (int): Number of steps (days) to forecast ahead.
        factors_modeled (list): List of factor names included in the VAR model.

    Returns:
        pd.DataFrame: DataFrame of simulated parameters [beta0, beta1, beta2, lambda_],
                      indexed by future business dates. Returns empty DataFrame on failure.
    """
    print(f"Generating deterministic factor forecast for {n_steps} steps...")
    if var_results is None:
        print("Error: Cannot forecast, VAR results object is None.")
        return pd.DataFrame()
    if factors_df.empty:
        print("Error: Cannot forecast, historical factors DataFrame is empty.")
        return pd.DataFrame()

    try:
        # Get the last observed values needed to start the forecast
        last_observation = factors_df[factors_modeled].values[-var_results.k_ar:]

        # Forecast n_steps ahead (deterministic point forecast)
        simulated_factor_values = var_results.forecast(y=last_observation, steps=n_steps)

        # Combine with the lambda parameter (use the last observed value, assumed constant)
        if 'lambda' in factors_df.columns:
            last_lambda = factors_df['lambda'].iloc[-1]
        else:
            print("Warning: 'lambda' column not found in factors_df. Using default value 1.0.")
            last_lambda = 1.0 # Or handle as error depending on requirements
        simulated_lambda = np.full((n_steps, 1), last_lambda)

        # Combine beta factors and lambda
        # Ensure order matches [beta0, beta1, beta2, lambda]
        simulated_params_array = np.hstack((simulated_factor_values, simulated_lambda))

        # Create a DataFrame for simulated parameters
        last_date = factors_df.index[-1]
        # Use 'B' frequency for business days
        sim_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_steps, freq='B')
        sim_params_df = pd.DataFrame(simulated_params_array, columns=['beta0', 'beta1', 'beta2', 'lambda'], index=sim_dates)

        print("Deterministic factor forecast complete.")
        return sim_params_df

    except Exception as e:
        print(f"Error during deterministic factor forecast: {e}")
        return pd.DataFrame()


def simulate_factors_stochastic(var_results, factors_df, n_steps=config.SIMULATION_STEPS, n_simulations=config.MC_RUNS, factors_modeled=['beta0', 'beta1', 'beta2']):
    """
    Generates multiple stochastic simulation paths for the factors using the fitted VAR model.

    Args:
        var_results (VARResultsWrapper): The fitted VAR model results.
        factors_df (pd.DataFrame): The historical factors DataFrame.
        n_steps (int): Number of steps (days) to simulate ahead for each path.
        n_simulations (int): Number of simulation paths to generate.
        factors_modeled (list): List of factor names included in the VAR model.

    Returns:
        np.array: A 3D numpy array of shape (n_simulations, n_steps, 4) containing
                  simulated parameters [beta0, beta1, beta2, lambda_] for each path.
                  Returns None on failure.
        pd.DatetimeIndex: Index of simulation dates.
    """
    print(f"Generating {n_simulations} stochastic factor simulations for {n_steps} steps...")
    if var_results is None or factors_df.empty:
        print("Error: Cannot simulate, VAR results or historical factors are missing.")
        return None, None

    try:
        k_ar = var_results.k_ar # Lag order
        k_factors = len(factors_modeled) # Number of factors in VAR (e.g., 3 for betas)
        intercept = var_results.params.iloc[0].values # Intercept term (k_factors,)
        coeffs = var_results.params.iloc[1:].values.T # Coefficient matrix (k_factors, k_factors * k_ar)
        sigma_u = var_results.sigma_u # Residual covariance matrix (k_factors, k_factors)
        # Cholesky decomposition for generating correlated random shocks
        chol_sigma = np.linalg.cholesky(sigma_u)

        # Get initial values (last k_ar observations)
        initial_values = factors_df[factors_modeled].values[-k_ar:] # Shape (k_ar, k_factors)

        # Prepare array to store all simulations
        # Shape: (n_simulations, n_steps + k_ar, k_factors) - store initial values too
        sim_values_all = np.zeros((n_simulations, n_steps + k_ar, k_factors))

        # Set initial values for all simulations
        sim_values_all[:, :k_ar, :] = initial_values

        # Generate simulations step-by-step
        for i in range(n_simulations):
            for t in range(k_ar, n_steps + k_ar):
                # Prepare lagged values vector (flattened history)
                # Shape (k_factors * k_ar,)
                lagged_vals = sim_values_all[i, t-k_ar:t, :].ravel() # Order matters: [y1(t-1), y2(t-1), ..., yk(t-1), y1(t-2), ...]

                # Generate random shocks N(0, sigma_u)
                random_shocks = chol_sigma @ np.random.normal(0, 1, size=k_factors) # Shape (k_factors,)

                # Calculate next step forecast (deterministic part + shock)
                # forecast = intercept + coeffs @ lagged_vals + random_shocks
                # Reshape coeffs if needed, ensure matrix multiplication aligns
                # Coeffs shape (k_factors, k_factors * k_ar), lagged_vals shape (k_factors * k_ar,)
                deterministic_part = intercept + coeffs @ lagged_vals
                sim_values_all[i, t, :] = deterministic_part + random_shocks

        # Extract the simulated future steps (excluding initial values)
        simulated_betas = sim_values_all[:, k_ar:, :] # Shape (n_simulations, n_steps, k_factors)

        # Add the lambda factor (assumed constant, using last historical value)
        if 'lambda' in factors_df.columns:
            last_lambda = factors_df['lambda'].iloc[-1]
        else:
            last_lambda = 1.0 # Default if missing
        # Create lambda array matching shape: (n_simulations, n_steps, 1)
        simulated_lambda = np.full((n_simulations, n_steps, 1), last_lambda)

        # Combine betas and lambda: result shape (n_simulations, n_steps, 4)
        simulated_params_full = np.concatenate((simulated_betas, simulated_lambda), axis=2)

        # Generate date index for simulations
        last_date = factors_df.index[-1]
        sim_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_steps, freq='B')

        print("Stochastic factor simulation complete.")
        return simulated_params_full, sim_dates

    except Exception as e:
        print(f"Error during stochastic factor simulation: {e}")
        # import traceback
        # print(traceback.format_exc())
        return None, None


# --- Yield Curve Generation ---

def generate_simulated_curves(simulated_params, maturities=config.MATURITIES_LIST):
    """
    Generates yield curves from simulated NS parameters.
    Handles both single path (2D DataFrame) and multiple paths (3D numpy array).

    Args:
        simulated_params (pd.DataFrame or np.array):
            - DataFrame (n_steps, 4) for single deterministic path.
            - 3D Numpy array (n_simulations, n_steps, 4) for stochastic paths.
        maturities (list or np.array): List of maturities (in years) for the yield curve.

    Returns:
        pd.DataFrame or np.array:
            - DataFrame (n_steps, n_maturities) for single path.
            - 3D Numpy array (n_simulations, n_steps, n_maturities) for multiple paths.
            Returns None on failure or if input format is wrong.
    """
    print("Generating simulated yield curves...")
    maturities_np = np.array(maturities)

    if isinstance(simulated_params, pd.DataFrame):
        # Handle single deterministic path (DataFrame input)
        simulated_curves = []
        for index, params_row in simulated_params.iterrows():
            yield_curve = nelson_siegel_yields(params_row.values, maturities_np)
            simulated_curves.append(yield_curve)
        curves_df = pd.DataFrame(simulated_curves, index=simulated_params.index, columns=maturities_np)
        print("Single simulated yield curve generation complete.")
        return curves_df

    elif isinstance(simulated_params, np.ndarray) and simulated_params.ndim == 3:
        # Handle multiple stochastic paths (3D numpy array input)
        n_simulations, n_steps, n_params = simulated_params.shape
        n_maturities = len(maturities_np)
        all_curves = np.zeros((n_simulations, n_steps, n_maturities))

        for i in range(n_simulations):
            for t in range(n_steps):
                params = simulated_params[i, t, :]
                all_curves[i, t, :] = nelson_siegel_yields(params, maturities_np)
            if (i + 1) % 100 == 0:
                 print(f"Generated curves for {i+1}/{n_simulations} simulations...")

        print("Multiple simulated yield curve generation complete.")
        return all_curves
    else:
        print("Error: Invalid input format for simulated_params. Expected DataFrame or 3D Numpy array.")
        return None


# --- Save/Load VAR Results ---

def save_var_results(var_results, path=config.VAR_RESULTS_PATH):
    """
    Saves the fitted VAR results object using pickle.

    Args:
        var_results (VARResultsWrapper): The fitted VAR results object.
        path (str): File path for saving the pickle file.

    Returns:
        bool: True if successful, False otherwise.
    """
    if var_results is None:
        print("Warning: VAR results object is None. Skipping save.")
        return False
    try:
        print(f"Saving VAR results to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(var_results, f)
        print("Save successful.")
        return True
    except Exception as e:
        print(f"Error saving VAR results to {path}: {e}")
        return False

def load_var_results(path=config.VAR_RESULTS_PATH):
    """
    Loads a fitted VAR results object from a pickle file.

    Args:
        path (str): File path for loading the pickle file.

    Returns:
        VARResultsWrapper: The loaded VAR results object, or None on failure.
    """
    try:
        print(f"Loading VAR results from {path}...")
        with open(path, 'rb') as f:
            var_results = pickle.load(f)
        print("Load successful.")
        # Optional: Add checks to verify it's the expected object type
        if not isinstance(var_results, VAR.results_class):
             print(f"Warning: Loaded object from {path} is not a VARResultsWrapper.")
             # return None # Or allow loading but warn
        return var_results
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading VAR results from {path}: {e}")
        return None