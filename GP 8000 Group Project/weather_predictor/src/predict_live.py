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

        print(f"\n{model_name.upper()} Model Prediction for {args.date}:")
        # Assuming order: max_temp, min_temp, max_wind
        print(f"  Max Temp: {model_prediction[0]:.2f}°C")
        print(f"  Min Temp: {model_prediction[1]:.2f}°C")
        print(f"  Max Wind: {model_prediction[2]:.2f} km/h")

        # Compare with live API data
        comparison_results = prediction_utils.compare_predictions_with_live_api(
            model_pred=model_prediction,
            date_str=args.date,
            target_mean=target_mean,
            target_std=target_std
        )

        all_results[model_name] = {
            "status": "success",
            "prediction": model_prediction.tolist(),
            "comparison": comparison_results
        }

    # --- 4. Print Final Summary (Optional) ---
    print("\n--- Prediction Summary ---")
    for model_name, result in all_results.items():
        print(f"\nModel: {model_name.upper()}")
        if result["status"] == "success":
            pred = result["prediction"]
            comp = result["comparison"]
            print(f"  Prediction: MaxT={pred[0]:.2f}, MinT={pred[1]:.2f}, Wind={pred[2]:.2f}")
            if comp:
                 print(f"  API Fetched: {comp.get('api_data_fetched', False)}")
                 if comp.get('api_general_forecast'):
                      api_temp = comp['details'].get('api_avg_temp')
                      api_wind = comp['details'].get('api_wind')
                      print(f"  API Forecast: AvgT={api_temp:.2f if api_temp is not None else 'N/A'}, HighWind={api_wind:.2f if api_wind is not None else 'N/A'}")
                 print(f"  Comparison (Temperature): {comp['comparison'].get('temperature', 'error')}")
                 print(f"  Comparison (Wind): {comp['comparison'].get('wind', 'error')}")
            else:
                 print("  Comparison data unavailable.")
        else:
            print(f"  Status: {result['status']} - {result['message']}")

    print("\n--- Live Prediction Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Weather and Compare with Live API")

    parser.add_argument('--model_type', type=str, required=True, choices=['transformer', 'mlp', 'all'],
                        help="Type of model to use for prediction ('transformer', 'mlp', or 'all')")
    parser.add_argument('--date', type=str, required=True,
                        help="Date for prediction in YYYY-MM-DD format")
    parser.add_argument('--models_base_dir', type=str, default=config.DEFAULT_MODELS_BASE_DIR,
                        help="Base directory where trained models and stats are saved")

    # Add optional argument to specify model paths directly? Might be complex.
    # Sticking to the base directory approach for now.

    args = parser.parse_args()

    main(args)
