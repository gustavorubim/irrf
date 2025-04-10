import sys
import os
import pandas as pd
import numpy as np
import time

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src import config
    from src.data_loader import load_data_from_parquet, save_data_to_parquet
    from src.nelson_siegel import (fit_historical_ns, load_ns_params_from_parquet,
                                   save_ns_params_to_parquet)
    from src.var_simulation import (fit_var_model, load_var_results, save_var_results,
                                    simulate_factors_stochastic, generate_simulated_curves)
    from src.portfolio import (define_portfolio, calculate_portfolio_value_timeseries,
                               calculate_drawdown)
    from src.plotting import (plot_beta_histograms, plot_portfolio_drawdown,
                              ensure_dir_exists)
except ImportError as e:
    print(f"Error importing modules. Make sure 'src' directory is in the Python path and all modules exist: {e}")
    sys.exit(1)

def main():
    """
    Main function to run Monte Carlo simulations and analysis.
    """
    print("--- Starting Monte Carlo Analysis ---")
    start_time = time.time()

    # Ensure necessary directories exist
    ensure_dir_exists(config.DATA_DIR)
    ensure_dir_exists(config.MC_DATA_DIR)
    ensure_dir_exists(config.PLOTS_DIR)

    # --- Step 1: Load Data and Fit Models (or Load Pre-computed) ---
    print("\nStep 1: Loading data and fitting models...")

    # Load Raw Data
    historical_yields = load_data_from_parquet(config.RAW_YIELDS_PATH)
    if historical_yields.empty:
        print(f"Error: Raw yield data file not found or empty at {config.RAW_YIELDS_PATH}.")
        print("Please run 'scripts/download_data.py' first.")
        sys.exit(1)
    maturities_vector = historical_yields.columns.to_numpy(dtype=float)

    # Fit/Load NS Factors
    # Option to load if exists, otherwise fit and save
    historical_ns_factors = load_ns_params_from_parquet(config.NS_PARAMS_PATH)
    if historical_ns_factors.empty:
        print("NS parameters not found, fitting now...")
        historical_ns_factors = fit_historical_ns(historical_yields)
        if historical_ns_factors.empty:
            print("Error: Nelson-Siegel fitting failed. Cannot proceed.")
            sys.exit(1)
        save_ns_params_to_parquet(historical_ns_factors, config.NS_PARAMS_PATH)
    else:
        print("Loaded pre-computed NS parameters.")

    # Fit/Load VAR Model
    var_results = load_var_results(config.VAR_RESULTS_PATH)
    if var_results is None:
        print("VAR results not found, fitting now...")
        var_results = fit_var_model(historical_ns_factors, maxlags=1, factors_to_model=['beta0', 'beta1', 'beta2'])
        if var_results is None:
             print("Error: VAR model fitting failed. Cannot proceed.")
             sys.exit(1)
        save_var_results(var_results, config.VAR_RESULTS_PATH)
    else:
        print("Loaded pre-computed VAR results.")

    # --- Step 2: Run Stochastic Simulations ---
    print(f"\nStep 2: Running {config.MC_RUNS} stochastic simulations...")
    simulated_params_3d, sim_dates = simulate_factors_stochastic(
        var_results,
        historical_ns_factors,
        n_steps=config.SIMULATION_STEPS,
        n_simulations=config.MC_RUNS
    )
    if simulated_params_3d is None:
        print("Error: Stochastic factor simulation failed.")
        sys.exit(1)
    print(f"Generated factor simulations. Shape: {simulated_params_3d.shape}") # (n_sim, n_steps, 4)

    # --- Step 3: Generate Simulated Yield Curves for All Paths ---
    # This can be memory intensive for large MC_RUNS and SIMULATION_STEPS
    print("\nStep 3: Generating yield curves for all simulations...")
    # Note: generate_simulated_curves returns a 3D numpy array here
    all_simulated_curves_3d = generate_simulated_curves(simulated_params_3d, maturities=maturities_vector)
    if all_simulated_curves_3d is None:
        print("Error: Generation of multiple yield curves failed.")
        sys.exit(1)
    print(f"Generated yield curve simulations. Shape: {all_simulated_curves_3d.shape}") # (n_sim, n_steps, n_mats)

    # Optional: Save all simulation paths (can be large!)
    # Consider saving in a more compressed format or only saving summaries if memory/disk is an issue.
    # For now, saving as Parquet requires converting to a 2D DataFrame with multi-index.
    # This might be inefficient. Saving the numpy array directly might be better if needed later.
    # print("\n(Optional) Saving all simulation paths...")
    # try:
    #     # Example: Save the numpy array directly
    #     np.save(config.MC_SIM_PATHS_PATH.replace('.parquet', '.npy'), all_simulated_curves_3d)
    #     print(f"Saved all simulation paths (numpy array) to {config.MC_SIM_PATHS_PATH.replace('.parquet', '.npy')}")
    # except Exception as e:
    #     print(f"Warning: Could not save all simulation paths: {e}")


    # --- Step 4: Portfolio Analysis ---
    print("\nStep 4: Performing portfolio analysis...")
    portfolio_def = define_portfolio() # Use default from config
    if portfolio_def is None:
        print("Error: Failed to define portfolio.")
        sys.exit(1)
    print(f"Using portfolio definition: Weights={portfolio_def['weights']}, Maturities={portfolio_def['maturities']}")

    max_drawdowns = []
    final_betas = [] # Collect final beta factors [b0, b1, b2] for each sim

    num_simulations = all_simulated_curves_3d.shape[0]
    for i in range(num_simulations):
        # Extract curves for simulation 'i' into a DataFrame
        curves_df = pd.DataFrame(all_simulated_curves_3d[i, :, :], index=sim_dates, columns=maturities_vector)

        # Calculate portfolio value time series for this simulation
        portfolio_ts = calculate_portfolio_value_timeseries(curves_df, portfolio_def)

        # Calculate drawdown
        drawdown_series, max_dd = calculate_drawdown(portfolio_ts)
        max_drawdowns.append(max_dd)

        # Collect final beta factors for this simulation
        # simulated_params_3d shape: (n_sim, n_steps, 4) -> [beta0, beta1, beta2, lambda]
        final_betas.append(simulated_params_3d[i, -1, :3]) # Get last step's beta0, beta1, beta2

        if (i + 1) % 100 == 0:
            print(f"Analyzed portfolio for {i + 1}/{num_simulations} simulations...")

    # --- Step 5: Save Analysis Results ---
    print("\nStep 5: Saving analysis results...")
    analysis_results = pd.DataFrame({
        'MaxDrawdown': max_drawdowns
    })
    # Create DataFrame for final betas
    final_betas_df = pd.DataFrame(final_betas, columns=['beta0', 'beta1', 'beta2'])
    # Combine results
    analysis_results = pd.concat([analysis_results, final_betas_df], axis=1)

    save_successful = save_data_to_parquet(analysis_results, config.MC_ANALYSIS_RESULTS_PATH)
    if save_successful:
        print(f"Analysis results saved to {config.MC_ANALYSIS_RESULTS_PATH}")
    else:
        print("Warning: Failed to save analysis results.")


    # --- Step 6: Generate Analysis Plots ---
    print("\nStep 6: Generating analysis plots...")

    # Plot Beta Histograms
    plot_beta_histograms(final_betas_df, output_path=config.BETA_HIST_PLOT_PATH)

    # Plot Drawdown Distribution
    plot_portfolio_drawdown(analysis_results['MaxDrawdown'], output_path=config.DRAWDOWN_PLOT_PATH)


    end_time = time.time()
    print(f"\n--- Monte Carlo Analysis Complete ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()