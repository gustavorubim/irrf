import pandas as pd
import numpy as np

# Import configuration for default portfolio definition
from . import config

def define_portfolio(weights=None, maturities=None):
    """
    Defines the hypothetical bond portfolio structure.
    Uses default from config if not provided.

    Args:
        weights (list, optional): List of weights for each bond in the portfolio.
                                  Defaults to config.PORTFOLIO_DEF['weights'].
        maturities (list, optional): List of maturities (in years) for each bond.
                                     Defaults to config.PORTFOLIO_DEF['maturities'].

    Returns:
        dict: A dictionary containing 'weights' and 'maturities' lists.
              Returns None if inputs are inconsistent or invalid.
    """
    if weights is None:
        weights = config.PORTFOLIO_DEF.get('weights', [])
    if maturities is None:
        maturities = config.PORTFOLIO_DEF.get('maturities', [])

    if not isinstance(weights, list) or not isinstance(maturities, list):
        print("Error: Portfolio weights and maturities must be lists.")
        return None
    if len(weights) != len(maturities):
        print("Error: Portfolio weights and maturities lists must have the same length.")
        return None
    if not weights or not maturities:
        print("Error: Portfolio definition cannot be empty.")
        return None
    if not np.isclose(sum(weights), 1.0):
        print(f"Warning: Portfolio weights do not sum to 1.0 (sum={sum(weights):.4f}). Normalizing.")
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]


    # Ensure maturities are floats
    try:
        maturities = [float(m) for m in maturities]
    except ValueError:
        print("Error: Portfolio maturities must be numeric.")
        return None

    return {'weights': weights, 'maturities': maturities}


def price_zero_coupon_bond(yield_rate, maturity, face_value=1.0):
    """
    Calculates the price of a zero-coupon bond.

    Args:
        yield_rate (float): The yield to maturity (annualized, decimal form, e.g., 0.05 for 5%).
        maturity (float): Time to maturity in years.
        face_value (float, optional): The face value of the bond. Defaults to 1.0.

    Returns:
        float: The calculated price of the bond. Returns NaN if maturity is non-positive.
    """
    if maturity <= 0:
        # Bond has matured or invalid maturity
        return np.nan # Or face_value if maturity is exactly 0? Depends on convention.
    # Simple discrete compounding formula: Price = FV / (1 + YTM)^T
    # Assumes yield_rate is already in the correct format (e.g., percentage points from Treasury data)
    # If yields are %, divide by 100: price = face_value / (1 + yield_rate / 100)**maturity
    # Assuming yield_rate is already a decimal (e.g., 5% is 5.0, not 0.05)
    price = face_value / (1 + yield_rate / 100)**maturity # Divide by 100 if yields are like 2.5, 3.0 etc.
    # If yields are already 0.025, 0.030 etc., use:
    # price = face_value / (1 + yield_rate)**maturity
    return price


def calculate_portfolio_value(yield_curve_series, portfolio_def, face_value_per_bond=1.0):
    """
    Calculates the value of the defined portfolio for a given yield curve.

    Args:
        yield_curve_series (pd.Series): A series representing the yield curve for a single date,
                                        where the index represents maturities (in years) and
                                        values are the corresponding yields (e.g., in %).
        portfolio_def (dict): Dictionary defining the portfolio {'weights': [...], 'maturities': [...]}.
        face_value_per_bond (float, optional): Assumed face value for each bond position
                                               corresponding to a weight. Defaults to 1.0.

    Returns:
        float: The total calculated value of the portfolio for that yield curve.
               Returns NaN if calculation fails (e.g., missing yield).
    """
    total_value = 0.0
    weights = portfolio_def['weights']
    maturities = portfolio_def['maturities']

    # Ensure yield curve index is numeric (float) for interpolation/lookup
    try:
        yield_curve_series.index = pd.to_numeric(yield_curve_series.index)
    except ValueError:
        print("Error: Yield curve index could not be converted to numeric maturities.")
        return np.nan

    for weight, maturity in zip(weights, maturities):
        # Find the yield for the specific maturity.
        # Option 1: Exact match (if available)
        # yield_rate = yield_curve_series.get(maturity, np.nan)

        # Option 2: Interpolation (linear is common for yield curves)
        # Need to handle cases where maturity is outside the range of the curve's index
        all_maturities = yield_curve_series.index.sort_values()
        if maturity < all_maturities.min() or maturity > all_maturities.max():
             print(f"Warning: Portfolio maturity {maturity} is outside the range of the yield curve [{all_maturities.min():.2f}, {all_maturities.max():.2f}]. Cannot interpolate/extrapolate accurately. Skipping bond.")
             # Or attempt extrapolation carefully, or return NaN for the portfolio value
             # For simplicity, we might skip this bond or return NaN
             return np.nan # Portfolio value is unreliable if yields are missing

        # Use numpy interpolation
        yield_rate = np.interp(maturity, yield_curve_series.index, yield_curve_series.values)


        if pd.isna(yield_rate):
            print(f"Warning: Could not find or interpolate yield for maturity {maturity}. Skipping bond.")
            # Decide how to handle missing yield: skip bond, return NaN for portfolio, etc.
            return np.nan # If any bond can't be priced, portfolio value is unreliable

        bond_price = price_zero_coupon_bond(yield_rate, maturity, face_value_per_bond)

        if pd.isna(bond_price):
             print(f"Warning: Could not price bond with maturity {maturity} and yield {yield_rate}. Skipping.")
             return np.nan

        # Value contribution = weight * price (assuming initial investment scaled weights)
        total_value += weight * bond_price

    return total_value


def calculate_portfolio_value_timeseries(simulated_curves_df, portfolio_def, face_value_per_bond=1.0):
    """
    Calculates the portfolio value over time for a given set of simulated yield curves.

    Args:
        simulated_curves_df (pd.DataFrame): DataFrame where rows are time steps (index)
                                           and columns are maturities (float years).
        portfolio_def (dict): Dictionary defining the portfolio.
        face_value_per_bond (float, optional): Assumed face value. Defaults to 1.0.

    Returns:
        pd.Series: Time series of portfolio values, indexed by date/time step.
                   Returns empty Series on failure.
    """
    if simulated_curves_df.empty or portfolio_def is None:
        print("Error: Cannot calculate portfolio value timeseries. Input data missing.")
        return pd.Series(dtype=float)

    portfolio_values = []
    for date, yield_curve_series in simulated_curves_df.iterrows():
        value = calculate_portfolio_value(yield_curve_series, portfolio_def, face_value_per_bond)
        portfolio_values.append(value)

    return pd.Series(portfolio_values, index=simulated_curves_df.index)


def calculate_drawdown(portfolio_value_timeseries):
    """
    Calculates the drawdown series and maximum drawdown from a portfolio value time series.

    Args:
        portfolio_value_timeseries (pd.Series): Time series of portfolio values.

    Returns:
        tuple:
            pd.Series: The drawdown series (percentage decline from the peak).
            float: The maximum drawdown experienced (maximum percentage decline).
            Returns (None, None) if input is invalid.
    """
    if not isinstance(portfolio_value_timeseries, pd.Series) or portfolio_value_timeseries.empty:
        print("Error: Input must be a non-empty pandas Series.")
        return None, None
    
    # Drop NaNs which can interfere with cummax()
    values = portfolio_value_timeseries.dropna()
    if values.empty:
        print("Warning: Portfolio value timeseries is all NaN after dropping NaNs.")
        return pd.Series(dtype=float), np.nan


    # Calculate cumulative maximum (peak) up to each point
    cumulative_max = values.cummax()

    # Calculate drawdown: (Cumulative Max - Current Value) / Cumulative Max
    # Represents the percentage loss from the previous peak
    drawdown_series = (cumulative_max - values) / cumulative_max

    # Maximum drawdown is the maximum value in the drawdown series
    max_drawdown = drawdown_series.max()

    # Return the full series (reindexed to original if needed) and the max value
    return drawdown_series.reindex(portfolio_value_timeseries.index), max_drawdown