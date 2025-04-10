import sys
import os

# Add the project root directory to the Python path
# This allows us to import modules from the 'src' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src import config
    from src.data_loader import download_treasury_yield_data, save_data_to_parquet
except ImportError as e:
    print(f"Error importing modules. Make sure 'src' directory is in the Python path: {e}")
    sys.exit(1)

def main():
    """
    Main function to download and save Treasury yield data.
    """
    print("--- Starting Data Download ---")

    # Ensure data directory exists
    if not os.path.exists(config.DATA_DIR):
        try:
            os.makedirs(config.DATA_DIR)
            print(f"Created data directory: {config.DATA_DIR}")
        except OSError as e:
            print(f"Error creating data directory {config.DATA_DIR}: {e}")
            sys.exit(1)

    # Download data
    raw_data = download_treasury_yield_data(config.DATA_URL)

    if raw_data is None or raw_data.empty:
        print("Failed to download or process data. Exiting.")
        sys.exit(1)
    else:
        print(f"Successfully downloaded and processed data. Shape: {raw_data.shape}")

    # Save data
    save_successful = save_data_to_parquet(raw_data, config.RAW_YIELDS_PATH)

    if save_successful:
        print(f"Raw yield data saved successfully to {config.RAW_YIELDS_PATH}")
        print("--- Data Download Complete ---")
    else:
        print("Failed to save raw yield data. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()