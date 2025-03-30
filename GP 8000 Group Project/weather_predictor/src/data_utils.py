import requests
import json
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Assuming config.py is in the same directory or accessible via PYTHONPATH
from . import config

def get_data_gov_sg_api(api_url, referer=config.API_REFERER):
    """
    Retrieves data from a Data.gov.sg API endpoint.

    Args:
        api_url (str): The URL of the Data.gov.sg API endpoint.
        referer (str): The referer header value.

    Returns:
        dict: A dictionary containing the API response data, or None if there was an error.
    """
    try:
        headers = {'referer': referer} if referer else {}
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API {api_url}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from {api_url}: {e}")
        print(f"Response text: {response.text[:500]}...") # Print first 500 chars of response
        return None


def download_dataset_file(dataset_id, referer=config.API_REFERER, max_polls=5, poll_interval=3):
    """
    Downloads a dataset file from Data.gov.sg using the initiate/poll mechanism.

    Args:
        dataset_id (str): The ID of the dataset to download.
        referer (str): The referer header value.
        max_polls (int): Maximum number of times to poll for the download URL.
        poll_interval (int): Seconds to wait between polls.

    Returns:
        pandas.DataFrame: The downloaded data as a DataFrame, or None if download fails.
    """
    s = requests.Session()
    s.headers.update({'referer': referer})
    initiate_url = f"{config.API_BASE_URL_OPEN}/v1/public/api/datasets/{dataset_id}/initiate-download"
    poll_url = f"{config.API_BASE_URL_OPEN}/v1/public/api/datasets/{dataset_id}/poll-download"

    try:
        # Initiate download
        initiate_response = s.get(initiate_url, headers={"Content-Type": "application/json"}, json={})
        initiate_response.raise_for_status()
        print(initiate_response.json().get('data', {}).get('message', 'Initiating download...'))

        # Poll for download URL
        for i in range(max_polls):
            print(f"Polling attempt {i+1}/{max_polls}...")
            poll_response = s.get(poll_url, headers={"Content-Type": "application/json"}, json={})
            poll_response.raise_for_status()
            poll_data = poll_response.json().get('data', {})
            print(f"Poll response data: {poll_data}")

            if "url" in poll_data:
                download_url = poll_data['url']
                print(f"Download URL obtained: {download_url}")
                try:
                    df = pd.read_csv(download_url)
                    print("\nDataFrame loaded successfully!")
                    print("First 5 rows:")
                    print(df.head())
                    return df
                except pd.errors.ParserError as e:
                    print(f"Error parsing CSV from {download_url}: {e}")
                    return None
                except Exception as e:
                    print(f"Error reading CSV from {download_url}: {e}")
                    return None

            if i == max_polls - 1:
                print(f"{i+1}/{max_polls}: No download URL found after maximum polls. Possible error with dataset.")
                print("Please try again or report the issue at https://go.gov.sg/datagov-supportform")
                return None
            else:
                print(f"{i+1}/{max_polls}: No result yet, continuing to poll in {poll_interval} seconds...\n")
                time.sleep(poll_interval)

    except requests.exceptions.RequestException as e:
        print(f"API request error during download process: {e}")
        if 'response' in locals() and poll_response:
             print(f"Last poll response status: {poll_response.status_code}")
             print(f"Last poll response text: {poll_response.text[:500]}...")
        elif 'initiate_response' in locals() and initiate_response:
             print(f"Initiate response status: {initiate_response.status_code}")
             print(f"Initiate response text: {initiate_response.text[:500]}...")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding error during download process: {e}")
        if 'response' in locals() and poll_response:
             print(f"Last poll response text: {poll_response.text[:500]}...")
        elif 'initiate_response' in locals() and initiate_response:
             print(f"Initiate response text: {initiate_response.text[:500]}...")
        return None
    finally:
        s.close() # Ensure session is closed

def add_cyclic_features(df):
    """
    Adds cyclic (sin/cos) features for day of year, day of week, and month
    to a DataFrame with a 'date' column.

    Args:
        df (pandas.DataFrame): DataFrame with a 'date' column (datetime objects).

    Returns:
        pandas.DataFrame: DataFrame with added cyclic features.
    """
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
         try:
            df['date'] = pd.to_datetime(df['date'])
         except Exception as e:
            raise ValueError(f"Could not convert 'date' column to datetime: {e}")

    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Create cyclic features
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    # Optionally drop intermediate columns if not needed
    # df = df.drop(columns=['day_of_year', 'day_of_week', 'month'])

    return df

# PyTorch Dataset class using cyclic features as input.
class WeatherDataset(Dataset):
    """
    PyTorch Dataset for weather prediction using cyclic date features.
    """
    def __init__(self, df, target_cols=['maximum_temperature', 'minimum_temperature', 'max_wind_speed']):
        """
        Args:
            df (pandas.DataFrame): DataFrame containing preprocessed data with cyclic features.
            target_cols (list): List of column names to be used as targets.
        """
        if not all(col in df.columns for col in target_cols):
            raise ValueError(f"DataFrame missing one or more target columns: {target_cols}")

        # Define feature columns (cyclic features)
        feature_cols = ['sin_day_of_year', 'cos_day_of_year',
                        'sin_day_of_week', 'cos_day_of_week',
                        'sin_month', 'cos_month']
        if not all(col in df.columns for col in feature_cols):
            raise ValueError(f"DataFrame missing one or more required cyclic feature columns: {feature_cols}")

        # Handle missing values in target columns
        df_clean = df.replace('na', np.nan) # Replace 'na' strings if present
        df_clean = df_clean.dropna(subset=target_cols)
        if len(df_clean) != len(df):
            print(f"Warning: Dropped {len(df) - len(df_clean)} rows with missing target values.")

        # Extract features and targets
        self.features = df_clean[feature_cols].values.astype(np.float32)
        self.targets = df_clean[target_cols].values.astype(np.float32)

        # Convert to tensors
        self.features = torch.tensor(self.features)  # shape: (N, 6)
        self.targets = torch.tensor(self.targets)    # shape: (N, num_targets)

        # Reshape features for Transformer: (N, seq_len=6, input_dim=1)
        # Each cyclic value is treated as a token/step in the sequence.
        self.features = self.features.unsqueeze(-1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    print("Testing data_utils module...")

    # Test API fetch
    print("\nTesting API fetch (24hr forecast)...")
    forecast_data = get_data_gov_sg_api(config.API_24HR_FORECAST_URL)
    if forecast_data and 'data' in forecast_data and 'records' in forecast_data['data'] and forecast_data['data']['records']:
        print("API fetch successful. Sample data:")
        print(json.dumps(forecast_data['data']['records'][0]['general'], indent=2))
    else:
        print("API fetch failed or returned unexpected data.")

    # Test dataset download
    print(f"\nTesting dataset download (ID: {config.DEFAULT_DATASET_ID})...")
    df_downloaded = download_dataset_file(config.DEFAULT_DATASET_ID)
    if df_downloaded is not None:
        print(f"Dataset downloaded successfully. Shape: {df_downloaded.shape}")

        # Test cyclic features
        print("\nTesting adding cyclic features...")
        try:
            df_cyclic = add_cyclic_features(df_downloaded.copy()) # Use copy to avoid modifying original
            print("Cyclic features added successfully. New columns:")
            print(df_cyclic[['date', 'sin_day_of_year', 'cos_day_of_year', 'sin_month', 'cos_month']].head())

            # Test WeatherDataset
            print("\nTesting WeatherDataset creation...")
            try:
                # Define target columns based on typical weather datasets
                # Adjust if your DEFAULT_DATASET_ID has different column names
                potential_target_cols = ['maximum_temperature', 'minimum_temperature', 'max_wind_speed',
                                         'mean_temperature', 'total_rainfall'] # Add more possibilities
                actual_target_cols = [col for col in potential_target_cols if col in df_cyclic.columns]

                if not actual_target_cols:
                     print("Could not find suitable target columns in the downloaded dataset for WeatherDataset test.")
                elif len(actual_target_cols) < 3:
                     print(f"Warning: Found only {len(actual_target_cols)} target columns: {actual_target_cols}. Using these.")
                     dataset = WeatherDataset(df_cyclic, target_cols=actual_target_cols)
                     print(f"WeatherDataset created. Size: {len(dataset)}")
                     features_sample, targets_sample = dataset[0]
                     print(f"Sample features shape: {features_sample.shape}") # Should be (6, 1)
                     print(f"Sample targets shape: {targets_sample.shape}") # Should be (num_targets,)
                else:
                     # Select the first 3 found target columns for consistency if more are available
                     selected_target_cols = actual_target_cols[:3]
                     print(f"Using target columns: {selected_target_cols}")
                     dataset = WeatherDataset(df_cyclic, target_cols=selected_target_cols)
                     print(f"WeatherDataset created. Size: {len(dataset)}")
                     features_sample, targets_sample = dataset[0]
                     print(f"Sample features shape: {features_sample.shape}") # Should be (6, 1)
                     print(f"Sample targets shape: {targets_sample.shape}") # Should be (3,)

            except Exception as e:
                print(f"Error creating WeatherDataset: {e}")

        except Exception as e:
            print(f"Error adding cyclic features: {e}")
    else:
        print("Dataset download failed. Skipping further tests dependent on data.")

    print("\nData utils module test finished.")
