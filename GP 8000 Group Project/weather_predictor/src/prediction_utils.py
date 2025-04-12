import torch
import numpy as np
import pandas as pd
import json

# Assuming data_utils.py and config.py are accessible
from . import data_utils
from . import config

def predict_weather(model, date_str):
    """
    Generates weather predictions for a given date using a loaded model.

    Computes cyclic features for the date and feeds them to the model.

    Args:
        model (torch.nn.Module): The pre-trained and loaded model instance (in eval mode).
        date_str (str): The date for prediction in 'YYYY-MM-DD' format.

    Returns:
        numpy.ndarray: A numpy array containing the model's predictions
                       (e.g., [max_temp, min_temp, max_wind]), or None if an error occurs.
    """
    try:
        date_obj = pd.to_datetime(date_str)
    except ValueError:
        print(f"Error: Invalid date format '{date_str}'. Please use YYYY-MM-DD.")
        return None

    # Compute cyclic features for the single date
    day_of_year = date_obj.dayofyear
    day_of_week = date_obj.dayofweek
    month = date_obj.month

    sin_day_of_year = np.sin(2 * np.pi * day_of_year / 365)
    cos_day_of_year = np.cos(2 * np.pi * day_of_year / 365)
    sin_day_of_week = np.sin(2 * np.pi * day_of_week / 7)
    cos_day_of_week = np.cos(2 * np.pi * day_of_week / 7)
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)

    # Create feature array (6 features)
    features_np = np.array([sin_day_of_year, cos_day_of_year,
                            sin_day_of_week, cos_day_of_week,
                            sin_month, cos_month], dtype=np.float32)

    # Reshape for model input: (batch=1, seq_len=6, input_dim=1)
    features = torch.tensor(features_np).unsqueeze(0).unsqueeze(-1)

    # Ensure model is in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model(features)

    # Return prediction as a numpy array
    return prediction.numpy()[0]


def compare_predictions_with_live_api(model_pred, date_str, target_mean, target_std, verbose=False):
    """
    Fetches live 24-hour forecast data for a given date and compares the
    extremeness of the model prediction with the live API forecast using z-scores.

    Args:
        model_pred (numpy.ndarray): The model's prediction array [max_temp, min_temp, max_wind].
        date_str (str): The date for which to fetch the live forecast ('YYYY-MM-DD').
                        Note: The API provides a forecast *starting* from a certain time,
                        so this comparison is approximate.
        target_mean (torch.Tensor or list/numpy array): Mean values [max_temp, min_temp, max_wind] from training data.
        target_std (torch.Tensor or list/numpy array): Std dev values [max_temp, min_temp, max_wind] from training data.
        verbose (bool): Whether to print detailed information. Default is False.

    Returns:
        dict: A dictionary containing comparison results:
              {
                  "api_data_fetched": bool,
                  "api_general_forecast": dict or None, # General section of API data
                  "comparison": {
                      "temperature": "model" or "api" or "equal" or "error",
                      "wind": "model" or "api" or "equal" or "error"
                  },
                  "details": {
                      "model_avg_temp": float, "api_avg_temp": float or None,
                      "model_temp_z": float, "api_temp_z": float or None,
                      "model_wind": float, "api_wind": float or None,
                      "model_wind_z": float, "api_wind_z": float or None
                  }
              }
              Returns None if model_pred is invalid or essential stats are missing.
    """
    if model_pred is None or len(model_pred) < 3:
        print("Error: Invalid model prediction provided for comparison.")
        return None
    if target_mean is None or target_std is None or len(target_mean) < 3 or len(target_std) < 3:
         print("Error: Missing or incomplete target mean/std statistics for comparison.")
         return None

    # Convert stats to numpy if they are tensors
    if isinstance(target_mean, torch.Tensor):
        target_mean = target_mean.numpy()
    if isinstance(target_std, torch.Tensor):
        target_std = target_std.numpy()

    # Only print introductory message if verbose mode is enabled
    if verbose:
        print(f"\n--- Comparing Model Prediction for {date_str} with Live API ---")
        print(f"Fetching live 24hr forecast from: {config.API_24HR_FORECAST_URL}")
    
    # Fetch live API data
    api_data = data_utils.get_data_gov_sg_api(config.API_24HR_FORECAST_URL)

    results = {
        "api_data_fetched": False,
        "api_general_forecast": None,
        "comparison": {"temperature": "error", "wind": "error"},
        "details": {
            "model_avg_temp": None, "api_avg_temp": None,
            "model_temp_z": None, "api_temp_z": None,
            "model_wind": None, "api_wind": None,
            "model_wind_z": None, "api_wind_z": None
        }
    }

    # --- Model Calculations ---
    model_max_temp, model_min_temp, model_wind = model_pred[0], model_pred[1], model_pred[2]
    model_avg_temp = (model_max_temp + model_min_temp) / 2.0
    results["details"]["model_avg_temp"] = model_avg_temp
    results["details"]["model_wind"] = model_wind

    # Reference stats for z-score calculation
    ref_temp_mean = (target_mean[0] + target_mean[1]) / 2.0
    ref_temp_std = (target_std[0] + target_std[1]) / 2.0 # Simplification: average std dev
    ref_wind_mean = target_mean[2]
    ref_wind_std = target_std[2]

    # Calculate model z-scores (handle potential division by zero in std)
    model_temp_z = abs(model_avg_temp - ref_temp_mean) / (ref_temp_std + 1e-8)
    model_wind_z = abs(model_wind - ref_wind_mean) / (ref_wind_std + 1e-8)
    results["details"]["model_temp_z"] = model_temp_z
    results["details"]["model_wind_z"] = model_wind_z

    # --- API Data Processing and Comparison ---
    if api_data and 'data' in api_data and 'records' in api_data['data'] and api_data['data']['records']:
        results["api_data_fetched"] = True
        general_forecast = api_data['data']['records'][0].get('general')
        results["api_general_forecast"] = general_forecast

        if general_forecast:
            if verbose:
                print("Live API data fetched successfully.")
                print("API General Forecast:", json.dumps(general_forecast, indent=2))

            try:
                # Temperature Comparison
                api_temp_low = float(general_forecast['temperature']['low'])
                api_temp_high = float(general_forecast['temperature']['high'])
                api_avg_temp = (api_temp_low + api_temp_high) / 2.0
                api_temp_z = abs(api_avg_temp - ref_temp_mean) / (ref_temp_std + 1e-8)

                results["details"]["api_avg_temp"] = api_avg_temp
                results["details"]["api_temp_z"] = api_temp_z

                if verbose:
                    print(f"\nTemperature - Model Avg: {model_avg_temp:.2f}°C (Z: {model_temp_z:.2f}), API Avg: {api_avg_temp:.2f}°C (Z: {api_temp_z:.2f})")
                
                if model_temp_z > api_temp_z:
                    if verbose:
                        print("  >> Model predicts a more extreme temperature forecast.")
                    results["comparison"]["temperature"] = "model"
                elif api_temp_z > model_temp_z:
                    if verbose:
                        print("  >> API predicts a more extreme temperature forecast.")
                    results["comparison"]["temperature"] = "api"
                else:
                    if verbose:
                        print("  >> Model and API predict similar temperature extremeness.")
                    results["comparison"]["temperature"] = "equal"

            except (KeyError, TypeError, ValueError) as e:
                if verbose:
                    print(f"  Warning: Could not process API temperature data for comparison: {e}")
                results["comparison"]["temperature"] = "error"

            try:
                # Wind Speed Comparison (comparing model max wind with API high wind)
                api_wind_high = float(general_forecast['wind']['speed']['high'])
                api_wind_z = abs(api_wind_high - ref_wind_mean) / (ref_wind_std + 1e-8)

                results["details"]["api_wind"] = api_wind_high
                results["details"]["api_wind_z"] = api_wind_z

                if verbose:
                    print(f"Wind Speed - Model Max: {model_wind:.2f} km/h (Z: {model_wind_z:.2f}), API High: {api_wind_high:.2f} km/h (Z: {api_wind_z:.2f})")
                
                if model_wind_z > api_wind_z:
                    if verbose:
                        print("  >> Model predicts a more extreme wind speed forecast.")
                    results["comparison"]["wind"] = "model"
                elif api_wind_z > model_wind_z:
                    if verbose:
                        print("  >> API predicts a more extreme wind speed forecast.")
                    results["comparison"]["wind"] = "api"
                else:
                    if verbose:
                        print("  >> Model and API predict similar wind speed extremeness.")
                    results["comparison"]["wind"] = "equal"

            except (KeyError, TypeError, ValueError) as e:
                if verbose:
                    print(f"  Warning: Could not process API wind speed data for comparison: {e}")
                results["comparison"]["wind"] = "error"
        else:
            if verbose:
                print("Warning: 'general' forecast data not found in the fetched API response.")
            results["comparison"]["temperature"] = "api_data_missing"
            results["comparison"]["wind"] = "api_data_missing"
    else:
        if verbose:
            print("Warning: Failed to fetch or parse live API data for comparison.")
        results["comparison"]["temperature"] = "api_fetch_failed"
        results["comparison"]["wind"] = "api_fetch_failed"

    if verbose:
        print("--- Comparison Finished ---")
    return results

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    print("Testing prediction_utils module...")

    # Mock model and stats for testing
    class MockModel(torch.nn.Module):
        def forward(self, x):
            # Return dummy predictions based on input shape
            batch_size = x.shape[0]
            # Simulate some variation, shape (batch, 3)
            return torch.randn(batch_size, 3) * 5 + torch.tensor([[30.0, 25.0, 15.0]])

    mock_model = MockModel()
    mock_target_mean = np.array([29.5, 24.5, 12.0])
    mock_target_std = np.array([2.5, 2.0, 5.0])
    sample_date = "2025-03-30" # Use a current/future date for relevant API forecast

    # Test predict_weather
    print(f"\nTesting predict_weather for date: {sample_date}")
    prediction = predict_weather(mock_model, sample_date)
    if prediction is not None:
        print(f"Model Prediction: Max Temp={prediction[0]:.2f}, Min Temp={prediction[1]:.2f}, Max Wind={prediction[2]:.2f}")

        # Test compare_predictions_with_live_api
        print(f"\nTesting comparison with live API for date: {sample_date}")
        comparison_result = compare_predictions_with_live_api(prediction, sample_date, mock_target_mean, mock_target_std, verbose=True)

        if comparison_result:
            print("\nComparison Result Summary:")
            print(json.dumps(comparison_result, indent=2))
        else:
            print("Comparison function returned None (error).")
    else:
        print("predict_weather function failed.")

    print("\nPrediction utils module test finished.")
