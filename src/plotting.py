import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Import configuration for default save paths
from . import config

def ensure_dir_exists(directory):
    """Creates the directory if it doesn't exist."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            # Depending on severity, might want to raise exception or return False

def plot_simulation_surface(simulated_curves_df, title="Simulated US Treasury Yield Curve Evolution", output_path=config.SIM_SURFACE_PLOT_PATH):
    """
    Plots the simulated yield curve evolution as a 3D surface.

    Args:
        simulated_curves_df (pd.DataFrame): DataFrame of simulated yield curves,
                                            index=time steps, columns=maturities.
        title (str): The title for the plot.
        output_path (str): Path to save the plot image file.
                           Defaults to config.SIM_SURFACE_PLOT_PATH.
    """
    print("Plotting simulation surface...")
    if simulated_curves_df.empty:
        print("No simulated data to plot.")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data for surface plot
    X = simulated_curves_df.columns.values # Maturities (should be numeric)
    # Ensure X is numeric
    try:
        X = pd.to_numeric(X)
    except ValueError:
        print("Error: Plotting requires numeric column headers (maturities).")
        plt.close(fig) # Close the figure if error occurs
        return

    Y = np.arange(len(simulated_curves_df.index)) # Time steps (as numbers)
    X, Y = np.meshgrid(X, Y)
    Z = simulated_curves_df.values # Yields

    # Create the surface plot
    try:
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    except Exception as e:
        print(f"Error during surface plot creation: {e}")
        plt.close(fig)
        return


    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel('Maturity (Years)')
    ax.set_ylabel('Simulation Time Steps (Days)')
    ax.set_zlabel('Yield (%)') # Assuming yields are in %

    # Customize Y ticks to show dates (optional, can get crowded)
    try:
        step = max(1, len(simulated_curves_df.index) // 6) # Show approx 6 date labels
        tick_indices = np.arange(0, len(simulated_curves_df.index), step)
        # Ensure index is DatetimeIndex or similar for strftime
        if isinstance(simulated_curves_df.index, pd.DatetimeIndex):
            tick_labels = [simulated_curves_df.index[i].strftime('%Y-%m-%d') for i in tick_indices]
            ax.set_yticks(tick_indices)
            ax.set_yticklabels(tick_labels, rotation=-15, ha='left', fontsize=8)
        else:
            # If index is not dates, just use step numbers
             ax.set_yticks(tick_indices)
             ax.set_yticklabels(tick_indices, fontsize=8) # Show step numbers

    except Exception as e:
        print(f"Warning: Could not set date labels on Y-axis: {e}")
        # Fallback to default numeric ticks if date formatting fails


    # Add a color bar
    try:
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Yield (%)')
    except Exception as e:
        print(f"Warning: Could not add colorbar: {e}")


    # Improve layout and save/show
    plt.tight_layout()
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        try:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        except Exception as e:
            print(f"Error saving plot to {output_path}: {e}")
    else:
        plt.show() # Show plot if no output path is provided

    plt.close(fig) # Close the figure object to free memory
    print("Surface plotting complete.")


def plot_beta_histograms(final_betas_df, output_path=config.BETA_HIST_PLOT_PATH):
    """
    Plots histograms for the distribution of final simulated beta factors.

    Args:
        final_betas_df (pd.DataFrame): DataFrame where each row represents the final
                                       beta factors [beta0, beta1, beta2] from one simulation run.
                                       Columns should be named 'beta0', 'beta1', 'beta2'.
        output_path (str): Path to save the plot image file.
                           Defaults to config.BETA_HIST_PLOT_PATH.
    """
    print("Plotting beta factor histograms...")
    if final_betas_df.empty or not all(col in final_betas_df.columns for col in ['beta0', 'beta1', 'beta2']):
        print("Error: Input DataFrame is empty or missing required beta columns ('beta0', 'beta1', 'beta2').")
        return

    n_factors = 3
    fig, axes = plt.subplots(1, n_factors, figsize=(15, 5), sharey=True)
    fig.suptitle('Distribution of Final Simulated NS Factors (Beta0, Beta1, Beta2)')

    factors = ['beta0', 'beta1', 'beta2']
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    for i, factor in enumerate(factors):
        ax = axes[i]
        data = final_betas_df[factor].dropna() # Drop NaNs before plotting
        if data.empty:
            print(f"Warning: No valid data for factor {factor}. Skipping histogram.")
            ax.set_title(f'{factor} (No Data)')
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue

        ax.hist(data, bins=30, color=colors[i], edgecolor='black')
        ax.set_title(f'Final {factor}')
        ax.set_xlabel('Value')
        if i == 0:
            ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        try:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        except Exception as e:
            print(f"Error saving plot to {output_path}: {e}")
    else:
        plt.show()

    plt.close(fig)
    print("Beta histogram plotting complete.")


def plot_portfolio_drawdown(drawdown_stats, title="Distribution of Maximum Portfolio Drawdown", output_path=config.DRAWDOWN_PLOT_PATH):
    """
    Plots a histogram of the maximum drawdown values from multiple simulations.

    Args:
        drawdown_stats (pd.Series or list or np.array): A collection of maximum drawdown values
                                                       (one value per simulation run).
        title (str): The title for the plot.
        output_path (str): Path to save the plot image file.
                           Defaults to config.DRAWDOWN_PLOT_PATH.
    """
    print("Plotting portfolio drawdown distribution...")
    # Convert input to pandas Series to easily handle NaNs and plotting
    if not isinstance(drawdown_stats, pd.Series):
        drawdown_stats = pd.Series(drawdown_stats)

    valid_drawdowns = drawdown_stats.dropna()

    if valid_drawdowns.empty:
        print("Error: No valid drawdown data provided for plotting.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid_drawdowns * 100, bins=30, color='salmon', edgecolor='black') # Plot as percentage
    ax.set_title(title)
    ax.set_xlabel('Maximum Drawdown (%)')
    ax.set_ylabel('Frequency (Number of Simulations)')
    ax.grid(axis='y', alpha=0.7)

    # Add some summary statistics to the plot
    mean_dd = valid_drawdowns.mean() * 100
    median_dd = valid_drawdowns.median() * 100
    ax.axvline(mean_dd, color='k', linestyle='dashed', linewidth=1)
    ax.axvline(median_dd, color='grey', linestyle='dotted', linewidth=1)
    min_ylim, max_ylim = ax.get_ylim()
    ax.text(mean_dd*1.1, max_ylim*0.9, f'Mean: {mean_dd:.2f}%', color='k')
    ax.text(median_dd*1.1, max_ylim*0.8, f'Median: {median_dd:.2f}%', color='grey')


    plt.tight_layout()

    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        try:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        except Exception as e:
            print(f"Error saving plot to {output_path}: {e}")
    else:
        plt.show()

    plt.close(fig)
    print("Drawdown plotting complete.")