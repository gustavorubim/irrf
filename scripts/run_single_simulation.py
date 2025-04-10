import sys
import os
import pandas as pd

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src import config
    from src.data_loader import load_data_from_parquet, save_data_to_parquet
    from src.nelson_siegel import fit_historical_ns, save_ns_params_to_parquet
    from src.var_simulation import (fit_var_model, save_var_results,
                                    forecast_factors_deterministic,
                                    generate_simulated_curves)
    from src.plotting import plot_simulation_surface, ensure_dir_exists
except ImportError as e:
    print(f"Error importing modules. Make sure 'src' directory is in the Python path and all modules exist: {e}")
    sys.exit(1)

def main():
    """
    Main function to run a single deterministic yield curve simulation.
    """
    print("--- Starting Single Simulation Run ---")

    # Ensure necessary directories exist
    ensure_dir_exists(config.DATA_DIR)
    ensure_dir_exists(config.PLOTS_DIR)

    # 1. Load Raw Data
    print("\nStep 1: Loading raw yield data...")
    historical_yields = load_data_from_parquet(config.RAW_YIELDS_PATH)
    if historical_yields.empty:
        print(f"Error: Raw yield data file not found or empty at {config.RAW_YIELDS_PATH}.")
        print("Please run 'scripts/download_data.py' first.")
        sys.exit(1)
    print(f"Loaded historical yields. Shape: {historical_yields.shape}")
    # Extract maturities vector from columns
    maturities_vector = historical_yields.columns.to_numpy(dtype=float)


    # 2. Fit Nelson-Siegel Historically
    print("\nStep 2: Fitting Nelson-Siegel model...")
    historical_ns_factors = fit_historical_ns(historical_yields)
    if historical_ns_factors.empty:
        print("Error: Nelson-Siegel fitting failed. Cannot proceed.")
        sys.exit(1)
    # Save NS parameters
    save_ns_params_to_parquet(historical_ns_factors, config.NS_PARAMS_PATH)


    # 3. Fit VAR Model
    print("\nStep 3: Fitting VAR model...")
    # Using VAR(1) for simplicity as in original script
    var_results = fit_var_model(historical_ns_factors, maxlags=1, factors_to_model=['beta0', 'beta1', 'beta2'])
    if var_results is None:
         print("Error: VAR model fitting failed. Cannot proceed.")
         sys.exit(1)
    # Save VAR results
    save_var_results(var_results, config.VAR_RESULTS_PATH)


    # 4. Simulate Factors (Deterministic Forecast)
    print("\nStep 4: Generating deterministic factor forecast...")
    simulated_factors = forecast_factors_deterministic(var_results, historical_ns_factors, n_steps=config.SIMULATION_STEPS)
    if simulated_factors.empty:
        print("Error: Factor simulation (deterministic forecast) failed. Cannot proceed.")
        sys.exit(1)


    # 5. Generate Simulated Yield Curves
    print("\nStep 5: Generating simulated yield curves...")
    simulated_yield_curves = generate_simulated_curves(simulated_factors, maturities=maturities_vector)
    if simulated_yield_curves is None or simulated_yield_curves.empty:
        print("Error: Simulated yield curve generation failed.")
        sys.exit(1)
    # Save simulated curves
    save_data_to_parquet(simulated_yield_curves, config.SINGLE_SIM_CURVES_PATH)


    # 6. Plot Simulation Surface
    print("\nStep 6: Plotting simulation results...")
    plot_simulation_surface(simulated_yield_curves,
                              title=f"Simulated US Treasury Yield Curve Evolution ({config.SIMULATION_YEARS} Years, Deterministic)",
                              output_path=config.SIM_SURFACE_PLOT_PATH)

    print("\n--- Single Simulation Run Complete ---")

if __name__ == "__main__":
    main()