import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import optuna

# Ensure the src directory is in the Python path
# This allows importing modules like config, data_utils etc.
# Adjust based on how you run the script (e.g., from the root 'weather_predictor' directory)
# If running from 'weather_predictor/src', this might not be needed.
# If running from 'weather_predictor', 'src.' prefix is needed for imports.
# Assuming running from 'weather_predictor' directory: python src/train.py ...
try:
    from src import config, data_utils, model_definitions, training_utils
except ImportError:
    # Fallback if running directly from src directory
    import config
    import data_utils
    import model_definitions
    import training_utils


def main(args):
    """Main function to handle the training workflow."""
    print("--- Starting Training Script ---")

    # --- 1. Load Data ---
    print(f"Loading dataset ID: {args.dataset_id}")
    df = data_utils.download_dataset_file(args.dataset_id, referer=config.API_REFERER)
    if df is None:
        print("Failed to download or load dataset. Exiting.")
        sys.exit(1)

    # --- 2. Preprocess Data ---
    print("Preprocessing data and adding cyclic features...")
    try:
        df_processed = data_utils.add_cyclic_features(df.copy())
    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)

    # Define target columns (ensure these exist in your dataset)
    # Make this configurable if datasets vary significantly
    target_cols = ['maximum_temperature', 'minimum_temperature', 'max_wind_speed']
    if not all(col in df_processed.columns for col in target_cols):
         print(f"Error: Dataset {args.dataset_id} is missing required target columns: {target_cols}")
         print(f"Available columns: {list(df_processed.columns)}")
         # Attempt to find alternatives or exit
         # Example: Try 'mean_temperature' if 'maximum_temperature' is missing
         # For now, we exit if the primary targets are missing.
         sys.exit(1)

    print(f"Using target columns: {target_cols}")
    dataset = data_utils.WeatherDataset(df_processed, target_cols=target_cols)
    print(f"Dataset created with {len(dataset)} samples.")
    if len(dataset) == 0:
        print("Error: Dataset is empty after processing (possibly all rows had NaNs in targets). Exiting.")
        sys.exit(1)

    # --- 3. Split Data ---
    print("Splitting data into training and validation sets (80/20 split)...")
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    print(f"Training set size: {len(train_subset)}, Validation set size: {len(val_subset)}")

    # --- 4. Calculate Stats from Training Data ---
    # Important: Calculate mean/std ONLY from the training subset to avoid data leakage
    train_targets = dataset.targets[train_idx]
    target_mean = train_targets.mean(dim=0)
    target_std = train_targets.std(dim=0)
    print(f"Calculated Training Target Mean: {target_mean.tolist()}")
    print(f"Calculated Training Target Std: {target_std.tolist()}")

    # --- 5. Determine Models to Train ---
    models_to_train = []
    if args.model_type == 'all':
        models_to_train = ['transformer', 'mlp']
    elif args.model_type in ['transformer', 'mlp']:
        models_to_train = [args.model_type]
    else:
        print(f"Error: Invalid model_type '{args.model_type}'. Choose 'transformer', 'mlp', or 'all'.")
        sys.exit(1)

    print(f"Models to train: {models_to_train}")

    # --- 6. Training Loop for Each Model ---
    for model_name in models_to_train:
        print(f"\n===== Processing Model: {model_name.upper()} =====")

        # --- 6a. Hyperparameter Tuning (Optional) ---
        best_hyperparams = {}
        if args.run_tuning:
            print(f"Running Optuna hyperparameter tuning for {model_name}...")
            study_name = f"{model_name}_tuning_{args.dataset_id}"
            storage_name = f"sqlite:///{study_name}.db" # Store results in a local SQLite DB
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name, # Use storage to resume if needed
                load_if_exists=True, # Load previous results if DB exists
                direction="minimize",
                pruner=optuna.pruners.MedianPruner()
            )

            objective_func = training_utils.create_optuna_objective(
                model_type=model_name,
                dataset=dataset, # Use full dataset for CV in Optuna
                target_mean=target_mean, # Use train stats for loss calculation consistency
                target_std=target_std,
                n_splits=5, # 5-fold CV
                train_epochs=20 # Short training for each trial fold
            )

            study.optimize(objective_func, n_trials=args.tuning_trials, timeout=args.tuning_timeout)

            print("Optuna Study Statistics: ")
            print(f"  Number of finished trials: {len(study.trials)}")
            print("  Best trial:")
            trial = study.best_trial
            best_hyperparams = trial.params
            print(f"    Value (Avg CV Loss): {trial.value:.4f}")
            print("    Params: ")
            for key, value in best_hyperparams.items():
                print(f"      {key}: {value}")
        else:
            print("Skipping hyperparameter tuning. Using default or pre-defined parameters.")
            # Define default hyperparameters here if not tuning
            if model_name == 'transformer':
                best_hyperparams = {'nhead': 4, 'emb_dim': 32, 'num_layers': 2, 'dropout': 0.1,
                                    'lr': 0.001, 'alpha': 0.5, 'penalty_factor': 2.0}
            elif model_name == 'mlp':
                 best_hyperparams = {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1,
                                     'lr': 0.001, 'alpha': 0.5, 'penalty_factor': 2.0}
            print(f"Using default hyperparameters: {best_hyperparams}")


        # --- 6b. Instantiate Model ---
        output_dim = len(target_cols)
        if model_name == 'transformer':
            # Ensure emb_dim is compatible with nhead from tuning results
            if best_hyperparams['emb_dim'] % best_hyperparams['nhead'] != 0:
                 print(f"Warning: Tuned emb_dim ({best_hyperparams['emb_dim']}) not divisible by nhead ({best_hyperparams['nhead']}). Adjusting...")
                 best_hyperparams['emb_dim'] = (best_hyperparams['emb_dim'] // best_hyperparams['nhead']) * best_hyperparams['nhead']
                 if best_hyperparams['emb_dim'] == 0: best_hyperparams['emb_dim'] = best_hyperparams['nhead']
                 print(f"Adjusted emb_dim: {best_hyperparams['emb_dim']}")

            model = model_definitions.WeatherTransformer(
                emb_dim=best_hyperparams['emb_dim'],
                nhead=best_hyperparams['nhead'],
                num_layers=best_hyperparams['num_layers'],
                dropout=best_hyperparams['dropout'],
                output_dim=output_dim
            )
            model_dir_name = config.TRANSFORMER_MODEL_DIR_NAME
            model_filename = config.TRANSFORMER_MODEL_FILENAME

        elif model_name == 'mlp':
            model = model_definitions.MLP(
                input_dim=6, # From cyclic features
                hidden_dim=best_hyperparams['hidden_dim'],
                num_layers=best_hyperparams['num_layers'],
                dropout=best_hyperparams['dropout'],
                output_dim=output_dim
            )
            model_dir_name = config.MLP_MODEL_DIR_NAME
            model_filename = config.MLP_MODEL_FILENAME

        # --- 6c. Define Save Paths ---
        model_save_path = config.get_model_path(args.models_base_dir, model_dir_name, model_filename)
        stats_save_path = config.get_stats_path(args.models_base_dir, model_dir_name, config.STATS_FILENAME)

        # --- 6d. Run Final Training ---
        training_utils.train_model(
            model=model,
            model_type_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            target_mean=target_mean,
            target_std=target_std,
            hyperparameters=best_hyperparams,
            num_epochs=args.epochs,
            model_save_path=model_save_path,
            stats_save_path=stats_save_path
        )

    print("\n--- Training Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Weather Prediction Models")

    parser.add_argument('--model_type', type=str, default='all', choices=['transformer', 'mlp', 'all'],
                        help="Type of model to train ('transformer', 'mlp', or 'all')")
    parser.add_argument('--dataset_id', type=str, default=config.DEFAULT_DATASET_ID,
                        help="Dataset ID from data.gov.sg to use for training")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of epochs for final training")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training and validation")
    parser.add_argument('--models_base_dir', type=str, default=config.DEFAULT_MODELS_BASE_DIR,
                        help="Base directory to save trained models and stats")

    # Optuna Tuning Arguments
    parser.add_argument('--run_tuning', action='store_true',
                        help="Flag to enable Optuna hyperparameter tuning before final training")
    parser.add_argument('--tuning_trials', type=int, default=20,
                        help="Number of trials for Optuna tuning (if enabled)")
    parser.add_argument('--tuning_timeout', type=int, default=None, # No timeout by default
                        help="Timeout in seconds for the Optuna study (if enabled)")

    args = parser.parse_args()

    # Create models base directory if it doesn't exist
    os.makedirs(args.models_base_dir, exist_ok=True)

    main(args)
