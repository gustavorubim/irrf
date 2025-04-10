import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from io import StringIO
import warnings
from datetime import datetime

# Import configuration constants
from . import config

# Suppress warnings for cleaner output (optional)
warnings.filterwarnings("ignore")

def download_treasury_yield_data(url=config.DATA_URL):
    """
    Downloads and parses Treasury yield curve XML data from the specified URL.

    Args:
        url (str): The URL to download the XML data from. Defaults to config.DATA_URL.

    Returns:
        pd.DataFrame: A DataFrame containing the historical yield curve data,
                      indexed by date, with maturities as columns (in years).
                      Returns an empty DataFrame if download or parsing fails.
    """
    print(f"Downloading data from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        print("Download successful.")

        # Parse XML
        xml_content = response.content
        # Find the namespace - it can change, so find it dynamically
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
             print(f"Error parsing XML: {e}")
             # Optionally save problematic content for debugging
             # with open("error_yield_curve.xml", "wb") as f:
             #     f.write(xml_content)
             return pd.DataFrame()

        # Namespace is usually defined like 'm' or 'd', find it in the root tag attributes
        # Handle cases where the root tag might not have the expected format
        namespace_uri = ''
        if '}' in root.tag and root.tag.startswith('{'):
            namespace_uri = root.tag.split('}')[0].strip('{')
        if not namespace_uri:
             print("Warning: Could not determine XML namespace automatically. Parsing might fail.")
             # Provide default or attempt common namespaces if needed, or return empty
             # For Treasury data, common namespaces might involve 'schemas.microsoft.com'
             # Example fallback (adjust if necessary):
             # namespace = {'ns': 'http://www.w3.org/2005/Atom', 'm': '...', 'd': '...'}
             return pd.DataFrame() # Or attempt parsing with assumed namespaces

        namespace = {'ns': namespace_uri}
        # Add common namespaces used within Treasury data properties
        namespace['m'] = 'http://schemas.microsoft.com/ado/2007/08/dataservices/metadata'
        namespace['d'] = 'http://schemas.microsoft.com/ado/2007/08/dataservices'


        data = []
        # Find all 'entry' elements, then 'content', then 'properties'
        for entry in root.findall('.//ns:entry', namespace):
            properties = entry.find('.//ns:content/m:properties', namespace)
            if properties is None: continue

            record = {}
            # Extract date - it's usually under a 'd:NEW_DATE' tag
            date_tag = properties.find('.//d:NEW_DATE', namespace)
            if date_tag is not None and date_tag.text:
                 # Date might have time component, strip it
                 try:
                     record['Date'] = pd.to_datetime(date_tag.text.split('T')[0])
                 except ValueError:
                     print(f"Warning: Could not parse date '{date_tag.text}'. Skipping record.")
                     continue # Skip record if date is invalid
            else:
                 continue # Skip if date is missing

            # Extract yields using MATURITY_MAP from config
            valid_record = True
            for key in config.MATURITY_MAP.keys():
                 yield_tag = properties.find(f'.//d:{key}', namespace)
                 if yield_tag is not None and yield_tag.text:
                     try:
                         record[key] = float(yield_tag.text)
                     except ValueError:
                         record[key] = np.nan # Handle non-numeric entries
                         # Decide if this makes the record invalid
                         # valid_record = False # Uncomment if any non-numeric yield invalidates row
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
        # Ensure all keys exist before selecting
        valid_keys = [key for key in config.MATURITY_MAP.keys() if key in df.columns]
        df = df[valid_keys]
        df.columns = [config.MATURITY_MAP[col] for col in df.columns]

        # Drop rows/cols with too many NaNs if necessary
        # Consider keeping rows with some NaNs if interpolation/forward fill is planned later
        df = df.dropna(axis=0, how='any') # Drop rows with ANY NaNs for simplicity for now
        print(f"Data processed. Shape after dropping NaNs: {df.shape}")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()
    except Exception as e:
        # Catch other potential errors during parsing or processing
        print(f"An unexpected error occurred during download/parsing: {e}")
        # Consider logging the traceback for debugging
        # import traceback
        # print(traceback.format_exc())
        return pd.DataFrame()


def save_data_to_parquet(df, path):
    """
    Saves a Pandas DataFrame to a Parquet file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The file path to save the Parquet file to.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if df.empty:
        print(f"Warning: DataFrame is empty. Skipping save to {path}.")
        return False
    try:
        print(f"Saving data to {path}...")
        df.to_parquet(path, index=True)
        print("Save successful.")
        return True
    except Exception as e:
        print(f"Error saving data to {path}: {e}")
        return False

def load_data_from_parquet(path):
    """
    Loads data from a Parquet file into a Pandas DataFrame.

    Args:
        path (str): The file path of the Parquet file.

    Returns:
        pd.DataFrame: The loaded DataFrame, or an empty DataFrame if loading fails.
    """
    try:
        print(f"Loading data from {path}...")
        df = pd.read_parquet(path)
        print(f"Load successful. Data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
        return pd.DataFrame()

# Example usage (optional, can be removed or put under if __name__ == "__main__")
# if __name__ == '__main__':
#     raw_data = download_treasury_yield_data()
#     if not raw_data.empty:
#         save_data_to_parquet(raw_data, config.RAW_YIELDS_PATH)
#         loaded_data = load_data_from_parquet(config.RAW_YIELDS_PATH)
#         print("\nLoaded data head:")
#         print(loaded_data.head())