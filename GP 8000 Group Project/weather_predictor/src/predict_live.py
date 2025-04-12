import argparse
import os
import sys
import json
import torch
import numpy as np
import pandas as pd # Required for predict_weather date conversion

# Ensure the src directory is in the Python path (similar logic as train.py)
try:
    from src import config, data_utils, model_definitions, prediction_utils, training_utils # training_utils needed for loading stats potentially
except ImportError:
    # Fallback if running directly from src directory
    import config
    import data_utils
    import model_definitions
    import prediction_utils
    # training_utils might not be strictly needed here unless loading hyperparameters from stats

def load_model_and_stats(model_type_name, models_base_dir):
    """Loads a trained model and its associated statistics."""
    print(f"--- Loading Model and Stats for: {model_type_name.upper()} ---")

    if model_type_name == 'transformer':
        model_dir_name = config.TRANSFORMER_MODEL_DIR_NAME
        model_filename = config.TRANSFORMER_MODEL_FILENAME
        ModelClass = model_definitions.WeatherTransformer
    elif model_type_name == 'mlp':
        model_dir_name = config.MLP_MODEL_DIR_NAME
        model_filename = config.MLP_MODEL_FILENAME
        ModelClass = model_definitions.MLP
    else:
        print(f"Error: Invalid model type '{model_type_name}' for loading.")
        return None, None, None

    model_path = config.get_model_path(models_base_dir, model_dir_name, model_filename)
    stats_path = config.get_stats_path(models_base_dir, model_dir_name, config.STATS_FILENAME)

    # Load Stats first to get hyperparameters needed for model instantiation
    if not os.path.exists(stats_path):
        print(f"Error: Statistics file not found at {stats_path}")
        return None, None, None

    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"Statistics loaded successfully from {stats_path}")
        hyperparams = stats.get("hyperparameters", {})
        target_mean = np.array(stats.get("target_mean"))
        target_std = np.array(stats.get("target_std"))
        output_dim = len(target_mean) # Determine output dim from stats

        if target_mean is None or target_std is None:
             print(f"Error: target_mean or target_std missing in stats file: {stats_path}")
             return None, None, None

    except (json.JSONDecodeError, IOError, KeyError, TypeError) as e:
        print(f"Error loading or parsing statistics file {stats_path}: {e}")
        return None, None, None

    # Instantiate Model based on loaded hyperparameters
    try:
        if model_type_name == 'transformer':
            # Ensure emb_dim and nhead are present and compatible
            if 'emb_dim' not in hyperparams or 'nhead' not in hyperparams:
                 raise KeyError("Transformer hyperparameters 'emb_dim' or 'nhead' missing in stats file.")
            if hyperparams['emb_dim'] % hyperparams['nhead'] != 0:
                 # This shouldn't happen if train.py saved correctly, but check anyway
                 print(f"Warning: emb_dim ({hyperparams['emb_dim']}) in stats not divisible by nhead ({hyperparams['nhead']}). Model loading might fail or behave unexpectedly.")
            model = ModelClass(
                emb_dim=hyperparams.get('emb_dim', 32), # Provide defaults just in case
                nhead=hyperparams.get('nhead', 4),
                num_layers=hyperparams.get('num_layers', 2),
                dropout=hyperparams.get('dropout', 0.1),
                output_dim=output_dim
            )
        elif model_type_name == 'mlp':
            model = ModelClass(
                input_dim=6, # Standard input dim
                hidden_dim=hyperparams.get('hidden_dim', 64),
                num_layers=hyperparams.get('num_layers', 2),
                dropout=hyperparams.get('dropout', 0.1),
                output_dim=output_dim
            )
    except KeyError as e:
         print(f"Error: Missing hyperparameter '{e}' in stats file needed for model instantiation.")
         return None, None, None
    except Exception as e:
         print(f"Error instantiating model {model_type_name}: {e}")
         return None, None, None


    # Load Model State Dictionary
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None, None

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Load to CPU
        model.eval() # Set model to evaluation mode
        print(f"Model state loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model state dictionary from {model_path}: {e}")
        return None, None, None

    return model, target_mean, target_std

def print_default_summary(all_results, date_str):
    """Prints a very concise summary showing only the most extreme predictions and forecast text."""
    print(f"\n--- Weather Forecast Summary for {date_str} ---")
    
    # Initialize variables to track extremes
    highest_max_temp = -float('inf')
    lowest_min_temp = float('inf')
    highest_max_wind = -float('inf')
    api_forecast_text = "N/A"
    api_data_available = False
    successful_models = []
    model_avg_temp = None
    api_avg_temp = None
    api_high_wind = None
    
    # Collect data across all models
    for model_name, result in all_results.items():
        if result["status"] == "success":
            successful_models.append(model_name)
            pred = result["prediction"]
            # Update predicted extremes
            highest_max_temp = max(highest_max_temp, pred[0])
            lowest_min_temp = min(lowest_min_temp, pred[1])
            highest_max_wind = max(highest_max_wind, pred[2])
            
            # Extract API data if available (use the first one we find)
            comp = result.get("comparison")
            if comp and comp.get('api_data_fetched'):
                api_data_available = True
                
                # Get forecast text if not already found
                if api_forecast_text == "N/A" and comp.get('api_general_forecast'):
                    forecast_data = comp.get('api_general_forecast', {})
                    api_forecast_text = forecast_data.get('forecast', {}).get('text', "N/A")
                    
                    # Get temperature and wind data
                    if 'temperature' in forecast_data:
                        temp_low = forecast_data['temperature'].get('low')
                        temp_high = forecast_data['temperature'].get('high')
                        if temp_low is not None and temp_high is not None:
                            api_avg_temp = (temp_low + temp_high) / 2.0
                    
                    if 'wind' in forecast_data and 'speed' in forecast_data['wind']:
                        if 'high' in forecast_data['wind']['speed']:
                            api_high_wind = forecast_data['wind']['speed']['high']

    if not successful_models:
        print("No successful model predictions were made.")
        return
        
    print(f"Models run: {', '.join(m.upper() for m in successful_models)}")
    
    # Calculate model average temperature
    if highest_max_temp > -float('inf') and lowest_min_temp < float('inf'):
        model_avg_temp = (highest_max_temp + lowest_min_temp) / 2.0
    
    # Determine which source has more extreme values
    if api_data_available:
        # For temperature
        if model_avg_temp is not None and api_avg_temp is not None:
            # Compare which is more extreme (further from 25°C reference)
            if abs(model_avg_temp - 25) > abs(api_avg_temp - 25):
                temp_source = "Model"
                temp_value = f"{model_avg_temp:.2f}°C"
            else:
                temp_source = "API" 
                temp_value = f"{api_avg_temp:.2f}°C"
            print(f"{temp_source} predicted a more extreme temperature of: {temp_value}")
            
        # For wind
        if highest_max_wind > -float('inf') and api_high_wind is not None:
            if highest_max_wind > api_high_wind:
                wind_source = "Model"
                wind_value = f"{highest_max_wind:.2f} km/h"
            else:
                wind_source = "API"
                wind_value = f"{api_high_wind} km/h"
            print(f"{wind_source} predicted a more extreme wind of: {wind_value}")
            
        # Show forecast
        print(f"\nForecast: {api_forecast_text}")
    else:
        print("Live API data could not be fetched or processed.")

def print_verbose_summary(all_results, date_str):
    """Prints a detailed, structured summary for each model."""
    print(f"\n--- Verbose Weather Prediction Summary for {date_str} ---")
    
    for model_name, result in all_results.items():
        print(f"\n=== Model: {model_name.upper()} ===")
        
        if result["status"] == "success":
            pred = result["prediction"]
            comp = result.get("comparison")
            
            print("  Status: Success")
            print("  Prediction:")
            print(f"    Max Temperature: {pred[0]:.2f}°C")
            print(f"    Min Temperature: {pred[1]:.2f}°C")
            print(f"    Max Wind Speed: {pred[2]:.2f} km/h")
            
            if comp:
                print("\n  API Comparison:")
                print(f"    API Data Fetched: {comp.get('api_data_fetched', False)}")
                
                if comp.get('api_data_fetched'):
                    # Print general forecast information
                    if comp.get('api_general_forecast'):
                        forecast = comp['api_general_forecast']
                        print("\n    API General Forecast:")
                        
                        # Temperature
                        if 'temperature' in forecast:
                            temp = forecast['temperature']
                            print(f"      Temperature: {temp.get('low')}°C to {temp.get('high')}°C")
                            avg_temp = (temp.get('low', 0) + temp.get('high', 0)) / 2.0
                            print(f"      Average Temperature: {avg_temp:.2f}°C")
                        
                        # Wind
                        if 'wind' in forecast:
                            wind = forecast['wind']
                            speed = wind.get('speed', {})
                            direction = wind.get('direction', 'N/A')
                            print(f"      Wind Speed: {speed.get('low', 'N/A')} to {speed.get('high', 'N/A')} km/h {direction}")
                        
                        # Forecast text and validity period
                        if 'forecast' in forecast:
                            print(f"      Forecast: {forecast['forecast'].get('text', 'N/A')} ({forecast['forecast'].get('code', 'N/A')})")
                        
                        if 'validPeriod' in forecast:
                            valid = forecast['validPeriod']
                            print(f"      Valid Period: {valid.get('text', 'N/A')}")
                            print(f"      Period Start: {valid.get('start', 'N/A')}")
                            print(f"      Period End: {valid.get('end', 'N/A')}")
                    
                    # Print detailed comparison results
                    if 'comparison' in comp:
                        comparison = comp['comparison']
                        print("\n    Comparison Results:")
                        print(f"      Temperature Comparison: {comparison.get('temperature', 'N/A')}")
                        print(f"      Wind Comparison: {comparison.get('wind', 'N/A')}")
                    
                    # Print detailed statistics
                    if 'details' in comp:
                        details = comp['details']
                        print("\n    Statistical Details:")
                        model_avg_temp = (pred[0] + pred[1]) / 2.0
                        print(f"      Model Avg Temperature: {model_avg_temp:.2f}°C")
                        print(f"      API Avg Temperature: {details.get('api_avg_temp', 'N/A')}")
                        print(f"      Model Wind Speed: {pred[2]:.2f} km/h")
                        print(f"      API Wind Speed: {details.get('api_wind', 'N/A')} km/h")
                        
                        if details.get('model_temp_z') is not None:
                            print(f"      Temperature Z-Score (Model): {details.get('model_temp_z', 'N/A'):.2f}")
                        if details.get('api_temp_z') is not None:
                            print(f"      Temperature Z-Score (API): {details.get('api_temp_z', 'N/A'):.2f}")
                        if details.get('model_wind_z') is not None:
                            print(f"      Wind Z-Score (Model): {details.get('model_wind_z', 'N/A'):.2f}")
                        if details.get('api_wind_z') is not None:
                            print(f"      Wind Z-Score (API): {details.get('api_wind_z', 'N/A'):.2f}")
                else:
                    print("    API data could not be fetched for comparison details.")
            else:
                print("  Comparison data unavailable.")
        else:
            print(f"  Status: {result['status']}")
            print(f"  Message: {result.get('message', 'No details')}")

def main(args):
    """Main function for live prediction and comparison."""
    print("--- Starting Live Prediction Script ---")

    # --- 1. Validate Date ---
    try:
        pd.to_datetime(args.date) # Validate date format early
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Please use YYYY-MM-DD.")
        sys.exit(1)

    # --- 2. Determine Models to Predict With ---
    models_to_predict = []
    if args.model_type == 'all':
        models_to_predict = ['transformer', 'mlp']
    elif args.model_type in ['transformer', 'mlp']:
        models_to_predict = [args.model_type]
    else:
        print(f"Error: Invalid model_type '{args.model_type}'. Choose 'transformer', 'mlp', or 'all'.")
        sys.exit(1)

    print(f"Models selected for prediction: {models_to_predict}")
    print(f"Prediction Date: {args.date}")
    if args.verbose:
        print("Verbose output enabled.")

    # --- 3. Prediction Loop ---
    all_results = {}
    for model_name in models_to_predict:
        model, target_mean, target_std = load_model_and_stats(model_name, args.models_base_dir)

        if model is None or target_mean is None or target_std is None:
            print(f"Skipping prediction for {model_name} due to loading errors.")
            all_results[model_name] = {"status": "error", "message": "Failed to load model or stats."}
            continue

        # Make prediction for the given date
        model_prediction = prediction_utils.predict_weather(model, args.date)

        if model_prediction is None:
            print(f"Prediction failed for {model_name}.")
            all_results[model_name] = {"status": "error", "message": "Prediction function failed."}
            continue

        # Only print intermediate results if verbose
        if args.verbose:
            print(f"\n{model_name.upper()} Model Prediction for {args.date}:")
            print(f"  Max Temp: {model_prediction[0]:.2f}°C")
            print(f"  Min Temp: {model_prediction[1]:.2f}°C")
            print(f"  Max Wind: {model_prediction[2]:.2f} km/h")
        else:
            print(f"Prediction successful for {model_name.upper()}.")

        # Compare with live API data - pass the verbose flag
        comparison_results = prediction_utils.compare_predictions_with_live_api(
            model_pred=model_prediction,
            date_str=args.date,
            target_mean=target_mean,
            target_std=target_std,
            verbose=args.verbose  # Pass the verbose flag here
        )

        all_results[model_name] = {
            "status": "success",
            "prediction": model_prediction.tolist(),
            "comparison": comparison_results
        }

    # --- 4. Print Final Summary based on output format preference ---
    if args.verbose:
        print_verbose_summary(all_results, args.date)
    else:
        print_default_summary(all_results, args.date)

    print("\n--- Live Prediction Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Weather and Compare with Live API")

    parser.add_argument('--model_type', type=str, required=True, choices=['transformer', 'mlp', 'all'],
                        help="Type of model to use for prediction ('transformer', 'mlp', or 'all')")
    parser.add_argument('--date', type=str, required=True,
                        help="Date for prediction in YYYY-MM-DD format")
    parser.add_argument('--models_base_dir', type=str, default=config.DEFAULT_MODELS_BASE_DIR,
                        help="Base directory where trained models and stats are saved")
    parser.add_argument('--verbose', action='store_true',
                        help="Display detailed output including all API data and comparison metrics")

    args = parser.parse_args()

    main(args)