import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import numpy as np
import optuna
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import json
import os
import math

# Assuming model_definitions.py and config.py are accessible
from . import model_definitions
from . import config

# --- Loss Function ---

def combined_loss(pred, target, target_mean, target_std, penalty_factor=2.0, alpha=0.5):
    """
    Combined loss: weighted sum of a custom loss (penalizing errors on extreme targets)
    and standard MSE loss.

    Args:
        pred (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth targets.
        target_mean (torch.Tensor): Mean of the target variables (from training data).
        target_std (torch.Tensor): Standard deviation of the target variables (from training data).
        penalty_factor (float): Multiplier for the penalty on extreme value errors.
        alpha (float): Weight for the custom loss component (0 <= alpha <= 1).
                       Weight for MSE loss will be (1 - alpha).

    Returns:
        torch.Tensor: The calculated combined loss value.
    """
    base_loss = (pred - target)**2
    # Create a mask for targets that are more than 1 std dev away from the mean
    mask = (torch.abs(target - target_mean) > target_std).float()
    # Custom loss component: MSE with added penalty for extreme values
    custom_loss_val = (base_loss * (1 + penalty_factor * mask)).mean()
    # Standard MSE loss component
    mse_loss_val = nn.MSELoss()(pred, target)
    # Weighted combination
    return alpha * custom_loss_val + (1 - alpha) * mse_loss_val

# --- Metrics ---

def compute_mape(pred, target):
    """
    Computes the Mean Absolute Percentage Error (MAPE).

    Args:
        pred (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth targets.

    Returns:
        float: The MAPE value.
    """
    epsilon = 1e-8  # Avoid division by zero
    return torch.mean(torch.abs((pred - target) / (target + epsilon))).item()

# --- Optuna Hyperparameter Tuning ---

def create_optuna_objective(model_type, dataset, target_mean, target_std, n_splits=5, train_epochs=20):
    """
    Creates the Optuna objective function for hyperparameter tuning, tailored
    to either the Transformer or MLP model.

    Args:
        model_type (str): 'transformer' or 'mlp'.
        dataset (torch.utils.data.Dataset): The full WeatherDataset instance.
        target_mean (torch.Tensor): Mean of the target variables.
        target_std (torch.Tensor): Standard deviation of the target variables.
        n_splits (int): Number of folds for K-Fold cross-validation.
        train_epochs (int): Number of epochs to train each model during a trial fold.

    Returns:
        function: The objective function for Optuna study.optimize().
    """

    def objective(trial):
        # --- Hyperparameter Sampling ---
        if model_type == 'transformer':
            # Sample nhead first, then compute proper bounds for emb_dim.
            nhead = trial.suggest_int("nhead", 2, 8, step=2) # Ensure even steps if needed
            # Ensure emb_dim is divisible by nhead
            min_emb_dim = max(16, nhead) # Lower bound for emb_dim
            max_emb_dim = 64
            # Suggest emb_dim that's a multiple of nhead within the range
            emb_dim = trial.suggest_int("emb_dim", min_emb_dim, max_emb_dim)
            emb_dim = (emb_dim // nhead) * nhead # Ensure divisibility
            if emb_dim == 0: emb_dim = nhead # Handle edge case if min_emb_dim < nhead

            num_layers = trial.suggest_int("num_layers", 1, 4)
            dropout = trial.suggest_float("dropout", 0.0, 0.3)
            output_dim = target_mean.shape[0] # Get output dim from stats

            model_params = {"emb_dim": emb_dim, "nhead": nhead, "num_layers": num_layers,
                            "dropout": dropout, "output_dim": output_dim}
            ModelClass = model_definitions.WeatherTransformer

        elif model_type == 'mlp':
            hidden_dim = trial.suggest_int("hidden_dim", 16, 128, step=16)
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            output_dim = target_mean.shape[0]
            input_dim = 6 # Assuming 6 cyclic features

            model_params = {"input_dim": input_dim, "hidden_dim": hidden_dim, "num_layers": num_layers,
                            "dropout": dropout, "output_dim": output_dim}
            ModelClass = model_definitions.MLP
        else:
            raise ValueError(f"Unsupported model_type for Optuna: {model_type}")

        # Common hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        alpha = trial.suggest_float("alpha", 0.3, 0.7) # Weight for custom loss
        penalty_factor = trial.suggest_float("penalty_factor", 1.0, 3.0) # Penalty for extreme errors

        # --- Cross-validation ---
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        val_losses = []

        for fold, (train_index, val_index) in enumerate(kf.split(np.arange(len(dataset)))):
            # print(f"  Optuna Trial Fold {fold+1}/{n_splits}")
            train_subset = Subset(dataset, train_index)
            val_subset = Subset(dataset, val_index)

            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

            # Instantiate new model for each fold
            model = ModelClass(**model_params)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Train for a fixed number of epochs for the trial
            for epoch in range(train_epochs):
                model.train()
                for features, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = combined_loss(outputs, targets, target_mean, target_std, penalty_factor, alpha)
                    loss.backward()
                    optimizer.step()

            # Evaluate on validation fold
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for features, targets in val_loader:
                    outputs = model(features)
                    loss = combined_loss(outputs, targets, target_mean, target_std, penalty_factor, alpha)
                    total_val_loss += loss.item() * features.size(0) # Accumulate loss scaled by batch size
            avg_val_loss = total_val_loss / len(val_subset)
            val_losses.append(avg_val_loss)

            # Optuna Pruning (optional but recommended for long trials)
            trial.report(avg_val_loss, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(val_losses) # Return average validation loss across folds

    return objective


# --- Main Training Loop ---

def train_model(model, model_type_name, train_loader, val_loader, target_mean, target_std, hyperparameters, num_epochs, model_save_path, stats_save_path):
    """
    Trains a model, plots metrics, saves the model and training statistics.

    Args:
        model (nn.Module): The model instance to train.
        model_type_name (str): Name of the model type (e.g., 'transformer', 'mlp') for printing.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        target_mean (torch.Tensor): Mean of the target variables (for loss calculation).
        target_std (torch.Tensor): Standard deviation of the target variables (for loss calculation).
        hyperparameters (dict): Dictionary of hyperparameters used (lr, alpha, penalty_factor, etc.).
        num_epochs (int): Number of epochs to train.
        model_save_path (str): Full path to save the trained model state dictionary (.pt).
        stats_save_path (str): Full path to save the training statistics (.json).

    Returns:
        tuple: (train_losses, val_losses, train_rmse, val_rmse, train_mape, val_mape)
               Lists containing metrics for each epoch.
    """
    print(f"\n--- Training {model_type_name} Model ---")
    print(f"Hyperparameters: {hyperparameters}")
    print(f"Saving model to: {model_save_path}")
    print(f"Saving stats to: {stats_save_path}")

    lr = hyperparameters.get('lr', 0.001) # Default LR if not provided
    alpha = hyperparameters.get('alpha', 0.5)
    penalty_factor = hyperparameters.get('penalty_factor', 2.0)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Ensure save directories exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(stats_save_path), exist_ok=True)

    # Containers for metrics
    train_losses, val_losses = [], []
    train_rmse, val_rmse = [], []
    train_mape, val_mape = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10 # Number of epochs to wait for improvement before stopping early

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_mape = 0.0
        num_train_samples = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = combined_loss(outputs, targets, target_mean, target_std, penalty_factor, alpha)
            loss.backward()
            optimizer.step()

            batch_size = features.size(0)
            running_loss += loss.item() * batch_size
            running_mape += compute_mape(outputs, targets) * batch_size
            num_train_samples += batch_size

        avg_train_loss = running_loss / num_train_samples
        avg_train_mape = running_mape / num_train_samples
        train_losses.append(avg_train_loss)
        train_rmse.append(np.sqrt(avg_train_loss)) # RMSE is sqrt of MSE component if alpha=0, otherwise approx
        train_mape.append(avg_train_mape)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        running_val_mape = 0.0
        num_val_samples = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = combined_loss(outputs, targets, target_mean, target_std, penalty_factor, alpha)

                batch_size = features.size(0)
                running_val_loss += loss.item() * batch_size
                running_val_mape += compute_mape(outputs, targets) * batch_size
                num_val_samples += batch_size

        avg_val_loss = running_val_loss / num_val_samples
        avg_val_mape = running_val_mape / num_val_samples
        val_losses.append(avg_val_loss)
        val_rmse.append(np.sqrt(avg_val_loss)) # Approx RMSE
        val_mape.append(avg_val_mape)

        # --- Logging ---
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f} (RMSE: {train_rmse[-1]:.4f}, MAPE: {train_mape[-1]*100:.2f}%)")
            print(f"  Val Loss:   {avg_val_loss:.4f} (RMSE: {val_rmse[-1]:.4f}, MAPE: {val_mape[-1]*100:.2f}%)")

        # --- Checkpoint Best Model & Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  (Checkpoint saved to {os.path.basename(model_save_path)} - Val Loss Improved)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss.")
                break

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")
    print(f"Model state dictionary saved to {model_save_path}")

    # --- Save Training Stats ---
    stats_data = {
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "hyperparameters": hyperparameters,
        "best_validation_loss": best_val_loss,
        "stopped_epoch": epoch + 1,
        # Optional: Add final metrics if needed
        # "final_train_loss": avg_train_loss,
        # "final_val_loss": avg_val_loss,
    }
    try:
        with open(stats_save_path, 'w') as f:
            json.dump(stats_data, f, indent=4)
        print(f"Training statistics saved to {stats_save_path}")
    except IOError as e:
        print(f"Error saving training statistics to {stats_save_path}: {e}")
    except TypeError as e:
         print(f"Error serializing statistics data to JSON: {e}")
         print(f"Data causing error: {stats_data}")


    # --- Plotting ---
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
    plt.plot(range(1, epoch + 2), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type_name} Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(1, epoch + 2), train_rmse, label='Train RMSE')
    plt.plot(range(1, epoch + 2), val_rmse, label='Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(f'{model_type_name} RMSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(1, epoch + 2), [m * 100 for m in train_mape], label='Train MAPE (%)')
    plt.plot(range(1, epoch + 2), [m * 100 for m in val_mape], label='Val MAPE (%)')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.title(f'{model_type_name} MAPE')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'{model_type_name} Training Metrics')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plot_filename = os.path.splitext(model_save_path)[0] + '_training_metrics.png'
    try:
        plt.savefig(plot_filename)
        print(f"Training metrics plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot to {plot_filename}: {e}")
    # plt.show() # Comment out or make optional for non-interactive runs

    return train_losses, val_losses, train_rmse, val_rmse, train_mape, val_mape
