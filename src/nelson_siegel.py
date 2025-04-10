import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings

# Import configuration constants and utility functions
from . import config
from .data_loader import save_data_to_parquet, load_data_from_parquet

# Suppress warnings for cleaner output (optional)
warnings.filterwarnings("ignore")

# --- Nelson-Siegel Model Functions ---

def nelson_siegel_yields(params, maturities):
    """
    Calculates yields using the Nelson-Siegel formula.

    Args:
        params (list or np.array): The NS parameters [beta0, beta1, beta2, lambda_].
        maturities (np.array): Array of maturities (in years).

    Returns:
        np.array: Array of calculated yields for the given maturities.
    """
    beta0, beta1, beta2, lambda_ = params
    # Ensure lambda_ is positive as required by the model derivation
    lambda_ = max(lambda_, 1e-6) # Prevent division by zero or negative lambda

    term1 = beta0
    # Use np.expm1 for better precision with small arguments: exp(x) - 1
    # (1 - exp(-lambda * T)) / (lambda * T)
    exp_term = np.exp(-lambda_ * maturities)
    term2_numerator = 1 - exp_term
    term2_denominator = lambda_ * maturities

    # Handle potential division by zero for maturity=0 or lambda=0
    # Limit as T->0: (1 - exp(-lambda*T))/(lambda*T) -> 1
    # Limit as lambda->0: (1 - exp(-lambda*T))/(lambda*T) -> T/T = 1 (using L'Hopital's rule)
    epsilon = 1e-8
    term2 = np.where(np.abs(term2_denominator) < epsilon,
                     beta1, # Limit is beta1 * 1
                     beta1 * term2_numerator / term2_denominator)

    # Term 3: beta2 * [ (1 - exp(-lambda*T))/(lambda*T) - exp(-lambda*T) ]
    term3_factor1 = np.where(np.abs(term2_denominator) < epsilon,
                             1.0, # Limit is 1
                             term2_numerator / term2_denominator)
    term3 = beta2 * (term3_factor1 - exp_term)

    return term1 + term2 + term3

def error_function(params, maturities, actual_yields):
    """
    Calculates the sum of squared errors for Nelson-Siegel fit.
    Includes penalty for non-positive lambda.

    Args:
        params (list or np.array): The NS parameters [beta0, beta1, beta2, lambda_].
        maturities (np.array): Array of maturities corresponding to actual_yields.
        actual_yields (np.array): Array of observed yields for a single date.

    Returns:
        float: The sum of squared errors, potentially penalized.
    """
    # Penalize non-positive lambda heavily
    if params[3] <= 0:
        return 1e10 # Return a large number

    model_yields = nelson_siegel_yields(params, maturities)
    # Ensure inputs are numpy arrays for vectorized operations
    actual_yields = np.asarray(actual_yields)
    model_yields = np.asarray(model_yields)
    error = np.sum((model_yields - actual_yields) ** 2)
    return error

def fit_nelson_siegel_on_date(maturities, yields_on_date):
    """
    Fits the NS model for a single date using optimization.

    Args:
        maturities (np.array): Array of maturities (in years).
        yields_on_date (np.array): Array of observed yields for the specific date.

    Returns:
        np.array: Fitted parameters [beta0, beta1, beta2, lambda_],
                  or np.full(4, np.nan) if optimization fails.
    """
    # Ensure inputs are numpy arrays
    maturities = np.asarray(maturities)
    yields_on_date = np.asarray(yields_on_date)

    # Filter out NaN yields and corresponding maturities for fitting
    valid_indices = ~np.isnan(yields_on_date)
    if not np.any(valid_indices):
        # print("Warning: All yields are NaN for this date. Cannot fit.")
        return np.full(4, np.nan)
    
    maturities_fit = maturities[valid_indices]
    yields_fit = yields_on_date[valid_indices]
    
    if len(yields_fit) < 4: # Need at least 4 points to fit 4 parameters
        # print(f"Warning: Not enough valid data points ({len(yields_fit)}) to fit NS model reliably.")
        return np.full(4, np.nan)


    # Initial guesses for parameters [beta0, beta1, beta2, lambda_]
    initial_lambda = 1.5
    # Use longest available yield for beta0 guess
    initial_beta0 = yields_fit[-1]
    # Use short-long spread for beta1 guess
    initial_beta1 = yields_fit[0] - yields_fit[-1]
    # Find mid-term maturity index for curvature guess (robust to missing maturities)
    target_mid_maturity = 2.0
    mid_maturity_idx = np.argmin(np.abs(maturities_fit - target_mid_maturity))
    # Calculate initial beta2 based on mid-point yield, beta0, beta1, lambda
    mid_maturity_val = maturities_fit[mid_maturity_idx]
    if abs(initial_lambda * mid_maturity_val) < 1e-8:
         initial_beta2 = 0 # Avoid division by zero if lambda*T is tiny
    else:
         term2_mid = initial_beta1 * (1 - np.exp(-initial_lambda * mid_maturity_val)) / (initial_lambda * mid_maturity_val)
         initial_beta2 = yields_fit[mid_maturity_idx] - (initial_beta0 + term2_mid)


    p0 = [initial_beta0, initial_beta1, initial_beta2, initial_lambda]

    # Bounds for parameters (lambda > 0 is crucial)
    bounds = [(None, None), (None, None), (None, None), (1e-6, None)] # lambda > 0

    try:
        result = minimize(error_function, p0, args=(maturities_fit, yields_fit),
                          method='L-BFGS-B', # Method that handles bounds
                          bounds=bounds)

        if result.success:
            # Optional: Check if lambda is reasonable (e.g., not excessively large)
            # if result.x[3] > 100: print(f"Warning: Fitted lambda is very large ({result.x[3]:.2f})")
            return result.x # Return the fitted parameters [beta0, beta1, beta2, lambda_]
        else:
            # print(f"Warning: Optimization failed for a date. Message: {result.message}")
            return np.full(4, np.nan) # Return NaNs if optimization fails
    except Exception as e:
        # print(f"Error during optimization: {e}")
        return np.full(4, np.nan)


def fit_historical_ns(yield_data):
    """
    Fits the NS model to each date in the historical yield data DataFrame.

    Args:
        yield_data (pd.DataFrame): DataFrame of historical yields, indexed by date,
                                   with maturities (years) as columns.

    Returns:
        pd.DataFrame: DataFrame of fitted NS parameters [beta0, beta1, beta2, lambda_],
                      indexed by date. Rows with fitting failures are dropped.
    """
    print("Fitting Nelson-Siegel model to historical data...")
    if yield_data.empty:
        print("Error: Input yield data is empty. Cannot fit NS model.")
        return pd.DataFrame()

    maturities = yield_data.columns.to_numpy(dtype=float)
    fitted_params = []
    dates = []

    # Use apply for potentially faster iteration (though loop is often clearer)
    # def fit_row(row):
    #     yields_on_date = row.to_numpy(dtype=float)
    #     return fit_nelson_siegel_on_date(maturities, yields_on_date)
    # fitted_params_list = yield_data.apply(fit_row, axis=1)
    # params_df = pd.DataFrame(fitted_params_list.tolist(), columns=['beta0', 'beta1', 'beta2', 'lambda'], index=yield_data.index)

    # Standard loop for clarity and easier debugging/printing warnings
    num_processed = 0
    num_failed = 0
    for date, row in yield_data.iterrows():
        yields_on_date = row.to_numpy(dtype=float)
        params = fit_nelson_siegel_on_date(maturities, yields_on_date)
        if not np.isnan(params).any():
            fitted_params.append(params)
            dates.append(date)
        else:
            # print(f"Skipping date {date} due to fitting issues or insufficient data.")
            num_failed += 1
        num_processed += 1
        if num_processed % 100 == 0:
             print(f"Processed {num_processed}/{len(yield_data)} dates...")


    if not fitted_params:
        print("Error: Nelson-Siegel fitting failed for all dates.")
        return pd.DataFrame()

    params_df = pd.DataFrame(fitted_params, columns=['beta0', 'beta1', 'beta2', 'lambda'], index=dates)
    # Drop any remaining rows where fitting might have produced NaNs (should be handled above)
    params_df = params_df.dropna()
    print(f"Nelson-Siegel fitting complete.")
    print(f"Successfully fitted {len(params_df)} dates. Failed/skipped {num_failed} dates.")
    print(f"Estimated factors shape: {params_df.shape}")
    return params_df


# --- Save/Load Functions for NS Parameters ---

def save_ns_params_to_parquet(params_df, path=config.NS_PARAMS_PATH):
    """
    Saves the DataFrame of fitted NS parameters to a Parquet file.

    Args:
        params_df (pd.DataFrame): DataFrame of NS parameters.
        path (str): File path for saving. Defaults to config.NS_PARAMS_PATH.

    Returns:
        bool: True if successful, False otherwise.
    """
    print(f"Saving NS parameters to {path}...")
    return save_data_to_parquet(params_df, path)

def load_ns_params_from_parquet(path=config.NS_PARAMS_PATH):
    """
    Loads the DataFrame of fitted NS parameters from a Parquet file.

    Args:
        path (str): File path for loading. Defaults to config.NS_PARAMS_PATH.

    Returns:
        pd.DataFrame: Loaded DataFrame of NS parameters, or empty DataFrame on failure.
    """
    print(f"Loading NS parameters from {path}...")
    return load_data_from_parquet(path)