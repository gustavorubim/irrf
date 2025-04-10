# Project Plan: IRRF Refactoring and Monte Carlo Analysis

**Overall Goal:** Transform the monolithic `irrf.py` into a modular Python project for simulating US Treasury yield curves, including data fetching, model fitting, single simulation runs, and a new Monte Carlo analysis component with portfolio drawdown and factor distribution analysis.

**Proposed Project Structure:**

```
irrf/
├── data/                 # Directory for storing intermediate and final data
│   ├── raw_yields.parquet
│   ├── ns_parameters.parquet
│   ├── var_parameters.pickle  # VAR results often saved as pickle
│   ├── single_simulation_curves.parquet
│   └── monte_carlo/
│       ├── simulation_paths.parquet # Store all 1000 paths
│       └── analysis_results.parquet
├── plots/                # Directory for saving generated plots
│   ├── simulation_surface.png
│   ├── beta_histograms.png
│   └── drawdown_analysis.png
├── scripts/              # Executable scripts for different workflow stages
│   ├── download_data.py
│   ├── run_single_simulation.py
│   └── run_monte_carlo_analysis.py
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── config.py         # Configuration and constants
│   ├── data_loader.py    # Data downloading and loading/saving
│   ├── nelson_siegel.py  # Nelson-Siegel model fitting and utilities
│   ├── var_simulation.py # VAR model fitting and simulation logic
│   ├── portfolio.py      # Hypothetical portfolio definition and valuation
│   └── plotting.py       # Plotting functions
├── notebooks/            # (Optional) Jupyter notebooks for exploration
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

**Detailed Plan Steps:**

1.  **Setup Project Structure:** Create the directories outlined above (`data`, `plots`, `scripts`, `src`, `notebooks` (optional)).
2.  **Create `src/config.py`:** Centralize constants like `DATA_URL`, `MATURITY_MAP`, `SIMULATION_YEARS`, `DAYS_PER_YEAR`, `SIMULATION_STEPS`, file paths (e.g., `RAW_DATA_PATH = "data/raw_yields.parquet"`), and potentially the number of Monte Carlo runs (`MC_RUNS = 1000`).
3.  **Refactor into `src` Modules:**
    *   **`src/data_loader.py`:** Move `download_treasury_yield_data`, add save/load functions (Parquet).
    *   **`src/nelson_siegel.py`:** Move NS functions, add save/load for parameters.
    *   **`src/var_simulation.py`:** Move VAR/simulation functions. Modify/add function for *stochastic* simulation using fitted VAR parameters and covariance for `MC_RUNS`. Add save/load for VAR results (pickle/Parquet).
    *   **`src/plotting.py`:** Move `plot_simulation_surface`. Add `plot_beta_histograms` and `plot_portfolio_drawdown`.
    *   **`src/portfolio.py`:** Define portfolio (e.g., 50% 2Y ZC, 50% 10Y ZC), add `calculate_portfolio_value` and `calculate_drawdown`.
4.  **Create `scripts`:**
    *   **`scripts/download_data.py`:** Imports from `src`, downloads, saves raw data.
    *   **`scripts/run_single_simulation.py`:** Imports from `src`, loads data, fits NS/VAR, runs *one* deterministic simulation, saves curve, saves plot.
    *   **`scripts/run_monte_carlo_analysis.py`:** Imports from `src`, loads data, fits NS/VAR *once*, runs `MC_RUNS` stochastic simulations, generates curves, saves all paths, defines portfolio, calculates values/drawdowns per path, collects results, saves analysis results, generates/saves analysis plots.
5.  **Create `README.md`:** Write project description, structure, setup, and usage instructions.
6.  **Create `requirements.txt`:** List dependencies (`pandas`, `numpy`, `requests`, `scipy`, `statsmodels`, `matplotlib`, `lxml`, `pyarrow`).

**Data Flow Diagram (Mermaid):**

```mermaid
graph LR
    subgraph User Interaction
        A[Start] --> B(Run download_data.py);
        B --> C{data/raw_yields.parquet};
        A --> D(Run run_single_simulation.py);
        C --> D;
        D --> E[plots/simulation_surface.png];
        D --> F[data/single_simulation_curves.parquet];
        A --> G(Run run_monte_carlo_analysis.py);
        C --> G;
        G --> H[data/monte_carlo/simulation_paths.parquet];
        G --> I[data/monte_carlo/analysis_results.parquet];
        G --> J[plots/beta_histograms.png];
        G --> K[plots/drawdown_analysis.png];
    end

    subgraph Modules Used
        M1[src/data_loader.py] --> B;
        M1 --> D;
        M1 --> G;
        M2[src/nelson_siegel.py] --> D;
        M2 --> G;
        M3[src/var_simulation.py] --> D;
        M3 --> G;
        M4[src/plotting.py] --> D;
        M4 --> G;
        M5[src/portfolio.py] --> G;
        M6[src/config.py] --> B & D & G;
    end

    style User Interaction fill:#f9f,stroke:#333,stroke-width:2px
    style Modules Used fill:#ccf,stroke:#333,stroke-width:2px