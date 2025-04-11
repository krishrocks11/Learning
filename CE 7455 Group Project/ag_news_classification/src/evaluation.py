import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve
)

def calculate_metrics(y_true, y_pred, labels=None):
    """
    Calculate evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional list of label names
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'labels': y_true
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path=None, class_names=None, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save the figure
        class_names: Optional list of class names
        title: Title for the plot
        
    Returns:
        fig: Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    if class_names:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Confusion matrix saved to {save_path}")
    
    return fig

def plot_training_history(history, save_path=None, title='Training History'):
    """
    Plot training and validation curves
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the figure
        title: Title for the plot
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training and validation loss
    if 'train_loss' in history and 'val_loss' in history:
        axs[0].plot(history['train_loss'], label='Training Loss')
        axs[0].plot(history['val_loss'], label='Validation Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training and Validation Loss')
        axs[0].legend()
    
    # Plot validation accuracy
    if 'val_accuracy' in history:
        axs[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Validation Accuracy')
        axs[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Training history saved to {save_path}")
    
    return fig

def compare_models(model_metrics, save_path=None, title='Model Comparison'):
    """
    Compare multiple models and plot results
    
    Args:
        model_metrics: Dictionary of model metrics dictionaries
        save_path: Path to save the figure
        title: Title for the plot
        
    Returns:
        fig: Matplotlib figure
    """
    models = list(model_metrics.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create comparison dataframe
    comparison = {metric: [] for metric in metrics}
    comparison['model'] = []
    
    for model in models:
        comparison['model'].append(model)
        for metric in metrics:
            if metric in model_metrics[model]:
                comparison[metric].append(model_metrics[model][metric])
            else:
                comparison[metric].append(0)
    
    comparison_df = pd.DataFrame(comparison)
    
    # Plot comparisons
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        sns.barplot(x='model', y=metric, data=comparison_df, ax=axs[row, col])
        axs[row, col].set_title(f'{metric.capitalize()} Comparison')
        axs[row, col].set_ylim(0, 1)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Model comparison saved to {save_path}")
    
    return fig, comparison_df

def export_results(metrics, report_path=None, format='csv'):
    """
    Export evaluation results to a file
    
    Args:
        metrics: Dictionary of evaluation metrics
        report_path: Path to save the report
        format: File format ('csv' or 'json')
        
    Returns:
        df: Pandas DataFrame of results
    """
    # Create a copy of metrics without large arrays
    results = {k: v for k, v in metrics.items() if not isinstance(v, (list, np.ndarray))}
    
    # Convert to DataFrame
    df = pd.DataFrame([results])
    
    if report_path:
        if format.lower() == 'csv':
            df.to_csv(report_path, index=False)
        elif format.lower() == 'json':
            df.to_json(report_path, orient='records')
        
        logging.info(f"Results exported to {report_path}")
    
    return df