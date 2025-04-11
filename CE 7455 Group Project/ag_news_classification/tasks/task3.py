import logging
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from src.data_loader import load_ag_news, create_data_loaders
from src.augmentation import apply_augmentations
from src.utils import get_device, set_seed, timer
from models.bert_classifier import BertClassifier, get_bert_tokenizer, train_bert, evaluate_bert
from models.bert_classifier import use_pretrained_bert
from models.naive_bayes import NaiveBayesClassifier
from models.neural_net import (TextCNN, TextRNN, build_vocab, create_neural_net_dataloaders,
                               train_neural_net, evaluate_neural_net)

def run(config, model_type=None):
    """
    Run Task 3: Impact of Dataset Size and Multiple Augmentations on Topic Classification
    
    Args:
        config: Configuration dictionary
        model_type: Specific model to run (if None, run all models)
    """
    logging.info("Starting Task 3: Impact of Dataset Size and Multiple Augmentations")
    
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Get device (CPU or GPU)
    device = get_device()
    
    # Define dataset sizes
    dataset_sizes = [0.1, 0.5, 1.0]  # 10%, 50%, 100%
    
    # Define augmentation combinations - removed backtranslation which was slow
    augmentation_combinations = [
        [],  # No augmentation (baseline)
        ['synonym'],  # Single augmentation
        ['deletion'],  # Single augmentation (replaced backtranslation)
        ['synonym', 'deletion'],  # Multiple augmentations
        ['synonym', 'deletion', 'swap']  # Multiple augmentations
    ]
    logging.info(f"Using augmentation combinations: {augmentation_combinations}")
    
    # Create results directory
    os.makedirs('results/task3', exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'model': [],
        'dataset_size': [],
        'augmentation': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'training_time': [],
        'inference_time': [],
        'loss': []
    }
    
    # Run experiments
    for size in dataset_sizes:
        logging.info(f"Running experiments with {size*100}% of dataset")
        
        # Load dataset with specified size
        data = load_ag_news(subset_size=size)
        
        for aug_combo in augmentation_combinations:
            aug_name = '_'.join(aug_combo) if aug_combo else 'baseline'
            logging.info(f"Running with augmentation: {aug_name}")
            
            # Apply augmentation if needed
            if aug_combo:
                augmented_data = augment_data(data, aug_combo)
            else:
                augmented_data = data
            
            # Run models
            if model_type is None or model_type == 'bert':
                with timer(f"BERT model (size={size}, aug={aug_name})"):
                    metrics = run_bert_model(augmented_data, device, config, size, aug_name)
                    add_results(results, 'bert', size, aug_name, metrics)
            
            if model_type is None or model_type == 'naive_bayes':
                with timer(f"Naive Bayes model (size={size}, aug={aug_name})"):
                    metrics = run_naive_bayes_model(augmented_data, config, size, aug_name)
                    add_results(results, 'naive_bayes', size, aug_name, metrics)
            
            if model_type is None or model_type == 'neural_net':
                with timer(f"Neural Network model (size={size}, aug={aug_name})"):
                    metrics = run_neural_net_model(augmented_data, device, config, size, aug_name)
                    add_results(results, 'neural_net', size, aug_name, metrics)
    
    # Create and save results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/task3/dataset_size_augmentation_results.csv', index=False)
    
    # Plot results
    plot_results(results_df)
    
    logging.info("Task 3 completed successfully")
    
    return results_df

def augment_data(data, techniques, multiplier=1):
    """
    Augment training data with specified techniques
    
    Args:
        data: Original data dictionary
        techniques: List of augmentation techniques
        multiplier: Number of augmented examples per original example
        
    Returns:
        augmented_data: Dictionary with augmented training data
    """
    train_texts, train_labels = data['train']
    
    logging.info(f"Augmenting training data with techniques: {techniques}")
    
    # Apply augmentation
    augmented_texts, augmented_labels = apply_augmentations(
        train_texts, train_labels, techniques, multiplier
    )
    
    # Create new data dictionary with augmented training data
    augmented_data = {
        'train': (augmented_texts, augmented_labels),
        'val': data['val'],
        'test': data['test']
    }
    
    return augmented_data

def add_results(results, model_name, dataset_size, augmentation, metrics):
    """
    Add model metrics to results dictionary
    
    Args:
        results: Results dictionary to update
        model_name: Name of the model
        dataset_size: Size of the dataset (0.1, 0.5, 1.0)
        augmentation: Augmentation technique(s) used
        metrics: Model metrics dictionary
    """
    results['model'].append(model_name)
    results['dataset_size'].append(dataset_size)
    results['augmentation'].append(augmentation)
    results['accuracy'].append(metrics['accuracy'])
    
    # Add precision, recall, F1 from classification metrics if available
    if 'precision' in metrics:
        results['precision'].append(metrics['precision'])
    else:
        # Calculate precision from predictions and labels
        precision, recall, f1, _ = precision_recall_fscore_support(
            metrics['labels'], metrics['predictions'], average='weighted'
        )
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
    
    if 'recall' in metrics and 'f1' in metrics:
        results['recall'].append(metrics['recall'])
        results['f1'].append(metrics['f1'])
    
    # Add training and inference time
    results['training_time'].append(metrics.get('training_time', 0))
    
    if 'avg_inference_time' in metrics:
        results['inference_time'].append(metrics['avg_inference_time'])
    else:
        results['inference_time'].append(metrics.get('inference_time', 0))
    
    # Add loss if available
    if 'val_loss' in metrics:
        results['loss'].append(metrics['val_loss'])
    else:
        results['loss'].append(0.0)

def run_bert_model(data, device, config, dataset_size, augmentation_name):
    """Use pre-trained AG News-specific BERT model for task 3"""
    prefix = f"bert_size{int(dataset_size*100)}_{augmentation_name}"
    
    logging.info(f"Using pre-trained AG News-specific BERT model with {dataset_size*100}% data and {augmentation_name} augmentation")
    
    bert_config = {
        'pretrained_model': 'lucasresck/bert-base-cased-ag-news',
        'num_classes': 4,
        'dropout_rate': 0.1,
        'save_model': True
    }
    
    # Update model save path for this specific task configuration
    bert_config['model_save_path'] = f'models/saved/{prefix}_model.pt'
    
    # Initialize tokenizer for AG News-specific model
    tokenizer = get_bert_tokenizer(pretrained_model=bert_config['pretrained_model'], use_ag_news_model=True)
    
    # Create data loaders
    dataloaders = create_data_loaders(data, tokenizer, batch_size=32)
    
    # Initialize model with the AG News-specific model
    model = BertClassifier(
        num_classes=bert_config['num_classes'],
        pretrained_model=bert_config['pretrained_model'],
        dropout_rate=bert_config['dropout_rate'],
        use_ag_news_model=True  # This is the key parameter to use the AG News model
    )
    
    # Use pre-trained model without training
    start_time = time.time()
    trained_model, history = use_pretrained_bert(model, dataloaders, device)
    training_time = time.time() - start_time
    
    # Evaluate model
    metrics = evaluate_bert(trained_model, dataloaders['test'], device)
    
    # Add additional metrics for task 3
    y_true = np.array(metrics['labels'])
    y_pred = np.array(metrics['predictions'])
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'dataset_size': dataset_size,
        'augmentation': augmentation_name
    })
    
    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'AG News-specific BERT Confusion Matrix\n{dataset_size*100}% Data, {augmentation_name} Augmentation')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(f'results/task3/bert_size{int(dataset_size*100)}_{augmentation_name}_cm.png')
    plt.close()
    
    # Save history data - Fix for DataFrame creation with values of different lengths
    # Convert each value to a list of the same length to avoid DataFrame error
    processed_history = {}
    for key, value in history.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            processed_history[key] = value
        else:
            # If not a sequence, convert to a single-item list
            processed_history[key] = [value]
    
    # Make sure all lists have the same length (pad if necessary)
    max_length = max(len(v) for v in processed_history.values())
    for key in processed_history:
        if len(processed_history[key]) < max_length:
            # Pad with the last value or None
            current_length = len(processed_history[key])
            if current_length > 0:
                pad_value = processed_history[key][-1]  # Use the last value for padding
            else:
                pad_value = None
            processed_history[key].extend([pad_value] * (max_length - current_length))
    
    # Save the processed history
    pd.DataFrame(processed_history).to_csv(f'results/task3/{prefix}_history.csv', index=False)
    
    return metrics

def run_naive_bayes_model(data, config, dataset_size, augmentation_name):
    """Run Naive Bayes model for task 3"""
    prefix = f"naive_bayes_size{int(dataset_size*100)}_{augmentation_name}"
    
    logging.info(f"Running Naive Bayes model with {dataset_size*100}% data and {augmentation_name} augmentation")
    
    # First try to load optimized Naive Bayes configuration
    nb_config = None
    try:
        with open('config/naive_bayes_config_optimized.yaml', 'r') as f:
            nb_config = yaml.safe_load(f)
            logging.info("Using optimized Naive Bayes configuration")
    except (FileNotFoundError, yaml.YAMLError) as e:
        logging.info(f"Optimized Naive Bayes config not found: {e}. Trying default config.")
    
    # If optimized config not found, try default config
    if nb_config is None:
        try:
            with open('config/naive_bayes_config.yaml', 'r') as f:
                nb_config = yaml.safe_load(f)
                logging.info("Using default Naive Bayes configuration")
        except Exception as e:
            logging.warning(f"Error loading Naive Bayes config: {e}. Using hardcoded defaults.")
            nb_config = {
                'vectorizer': 'tfidf',
                'max_features': 10000,
                'stop_words': 'english',
                'ngram_range': [1, 2],
                'alpha': 1.0
            }
    
    # Initialize model
    model = NaiveBayesClassifier(nb_config)
    
    # Train model
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['val']
    test_texts, test_labels = data['test']
    
    history = model.train(train_texts, train_labels, val_texts, val_labels)
    
    # Evaluate model
    metrics = model.evaluate(test_texts, test_labels)
    
    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, metrics['predictions'], average='weighted'
    )
    
    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    # Save metrics
    pd.DataFrame({k: [v] for k, v in history.items()}).to_csv(
        f'results/task3/{prefix}_history.csv', index=False
    )
    
    # Create confusion matrix
    cm = confusion_matrix(test_labels, metrics['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Naive Bayes Confusion Matrix (Size: {dataset_size*100}%, Aug: {augmentation_name})')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(f'results/task3/{prefix}_confusion_matrix.png')
    plt.close()
    
    return metrics

def run_neural_net_model(data, device, config, dataset_size, augmentation_name):
    """Run Neural Network model for task 3"""
    # First try to load optimized Neural Network configurations (CNN or RNN)
    nn_config = None
    
    # Try CNN optimized config first
    try:
        with open('config/neural_net_cnn_config_optimized.yaml', 'r') as f:
            nn_config = yaml.safe_load(f)
            logging.info("Using optimized CNN configuration")
    except (FileNotFoundError, yaml.YAMLError) as e:
        logging.info(f"Optimized CNN config not found: {e}. Trying RNN config.")
    
    # If CNN config not found, try RNN optimized config
    if nn_config is None:
        try:
            with open('config/neural_net_rnn_config_optimized.yaml', 'r') as f:
                nn_config = yaml.safe_load(f)
                logging.info("Using optimized RNN configuration")
        except (FileNotFoundError, yaml.YAMLError) as e:
            logging.info(f"Optimized RNN config not found: {e}. Trying default config.")
    
    # If no optimized config found, try default config
    if nn_config is None:
        try:
            with open('config/neural_net_config.yaml', 'r') as f:
                nn_config = yaml.safe_load(f)
                logging.info("Using default Neural Network configuration")
        except Exception as e:
            logging.warning(f"Error loading Neural Network config: {e}. Using hardcoded defaults.")
            nn_config = {
                'model_type': 'cnn',  # 'cnn' or 'rnn'
                'vocab_size': 50000,
                'embedding_dim': 300,
                'max_length': 100,
                'min_freq': 2,
                # CNN-specific
                'num_filters': 100,
                'filter_sizes': [3, 4, 5],
                # RNN-specific
                'hidden_dim': 256,
                'num_layers': 2,
                'bidirectional': True,
                'cell_type': 'lstm',  # 'lstm' or 'gru'
                # Training
                'batch_size': 64,
                'learning_rate': 0.001,
                'num_epochs': 10,
                'dropout_rate': 0.5,
                'save_model': True
            }
    
    # Set task-specific prefix and model save path
    prefix = f"{nn_config['model_type']}_size{int(dataset_size*100)}_{augmentation_name}"
    nn_config['model_save_path'] = f'models/saved/{prefix}_model.pt'
    
    # Ensure required parameters are present with defaults if not found
    if 'num_classes' not in nn_config:
        nn_config['num_classes'] = 4  # AG News has 4 classes

    # Ensure min_freq is present
    if 'min_freq' not in nn_config:
        nn_config['min_freq'] = 2
        
    logging.info(f"Running {nn_config['model_type'].upper()} model with {dataset_size*100}% data and {augmentation_name} augmentation")
    
    train_texts, train_labels = data['train']
    
    # Build vocabulary
    vocab, word_to_idx = build_vocab(train_texts, nn_config['min_freq'])
    nn_config['vocab_size'] = len(vocab)
    
    # Create dataloaders
    dataloaders = create_neural_net_dataloaders(
        data, word_to_idx, nn_config['batch_size'], nn_config['max_length']
    )
    
    # Initialize model
    if nn_config['model_type'].lower() == 'cnn':
        model = TextCNN(
            vocab_size=nn_config['vocab_size'],
            embedding_dim=nn_config['embedding_dim'],
            num_filters=nn_config['num_filters'],
            filter_sizes=nn_config['filter_sizes'],
            num_classes=nn_config.get('num_classes', 4),  # Use get with default
            dropout_rate=nn_config['dropout_rate']
        )
    else:  # RNN, LSTM, GRU
        model = TextRNN(
            vocab_size=nn_config['vocab_size'],
            embedding_dim=nn_config['embedding_dim'],
            hidden_dim=nn_config['hidden_dim'],
            num_classes=nn_config.get('num_classes', 4),  # Use get with default
            num_layers=nn_config['num_layers'],
            bidirectional=nn_config['bidirectional'],
            dropout_rate=nn_config['dropout_rate'],
            cell_type=nn_config['cell_type']
        )
    
    # Train model
    start_time = time.time()
    trained_model, history = train_neural_net(model, dataloaders, device, nn_config)
    training_time = time.time() - start_time
    
    # Evaluate model
    metrics = evaluate_neural_net(trained_model, dataloaders['test'], device)
    metrics['training_time'] = training_time
    
    # Add validation loss from history
    if history and 'val_loss' in history and len(history['val_loss']) > 0:
        metrics['val_loss'] = history['val_loss'][-1]
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        metrics['labels'], metrics['predictions'], average='weighted'
    )
    
    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    # Save training history
    pd.DataFrame(history).to_csv(f'results/task3/{prefix}_history.csv', index=False)
    
    # Plot and save training curves
    plot_training_curves(history, f'results/task3/{prefix}_training_curves.png')
    
    # Create confusion matrix
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{nn_config["model_type"].upper()} Confusion Matrix (Size: {dataset_size*100}%, Aug: {augmentation_name})')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(f'results/task3/{prefix}_confusion_matrix.png')
    plt.close()
    
    return metrics

def plot_training_curves(history, save_path):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_results(results_df):
    """
    Create visualizations for dataset size and augmentation results
    
    Args:
        results_df: DataFrame with experimental results
    """
    # Create output directory
    os.makedirs('results/task3/plots', exist_ok=True)
    
    # 1. Plot accuracy by dataset size and model
    plt.figure(figsize=(12, 8))
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        sizes = sorted(model_data['dataset_size'].unique())
        accuracies = []
        
        for size in sizes:
            # Get baseline accuracy for this size and model
            baseline_acc = model_data[(model_data['dataset_size'] == size) & 
                                     (model_data['augmentation'] == 'baseline')]['accuracy'].values[0]
            accuracies.append(baseline_acc)
        
        plt.plot([s*100 for s in sizes], accuracies, marker='o', label=model)
    
    plt.xlabel('Dataset Size (%)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Dataset Size (Baseline - No Augmentation)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/task3/plots/accuracy_vs_dataset_size.png')
    plt.close()
    
    # 2. Plot augmentation impact for each model at each dataset size
    for size in sorted(results_df['dataset_size'].unique()):
        size_data = results_df[results_df['dataset_size'] == size]
        
        plt.figure(figsize=(14, 10))
        
        # Create grouped bar chart
        aug_names = sorted(size_data['augmentation'].unique())
        x = np.arange(len(aug_names))
        width = 0.25
        
        for i, model in enumerate(sorted(size_data['model'].unique())):
            model_accs = []
            for aug in aug_names:
                acc = size_data[(size_data['model'] == model) & 
                               (size_data['augmentation'] == aug)]['accuracy'].values[0]
                model_accs.append(acc)
            
            plt.bar(x + (i-1)*width, model_accs, width, label=model)
        
        plt.xlabel('Augmentation Technique')
        plt.ylabel('Accuracy')
        plt.title(f'Impact of Augmentations at {int(size*100)}% Dataset Size')
        plt.xticks(x, aug_names)
        plt.legend()
        plt.grid(True, axis='y')
        plt.savefig(f'results/task3/plots/augmentation_impact_size{int(size*100)}.png')
        plt.close()
    
    # 3. Create heatmap of accuracy by model, dataset size and augmentation
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        
        # Create pivot table for heatmap
        pivot = model_data.pivot(index='dataset_size', columns='augmentation', values='accuracy')
        
        # Convert index to percentages
        pivot.index = [f"{int(idx*100)}%" for idx in pivot.index]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, cmap='Blues', fmt='.4f')
        plt.title(f'{model} Accuracy by Dataset Size and Augmentation')
        plt.savefig(f'results/task3/plots/{model}_heatmap.png')
        plt.close()
    
    # 4. Plot training time for different models and sizes
    plt.figure(figsize=(12, 8))
    
    for model in results_df['model'].unique():
        model_data = results_df[(results_df['model'] == model) & 
                               (results_df['augmentation'] == 'baseline')]
        sizes = [s*100 for s in sorted(model_data['dataset_size'].unique())]
        times = model_data.sort_values('dataset_size')['training_time'].values
        
        plt.plot(sizes, times, marker='o', label=model)
    
    plt.xlabel('Dataset Size (%)')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs. Dataset Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/task3/plots/training_time_vs_dataset_size.png')
    plt.close()
    
    # 5. Scatter plot of accuracy vs training time
    plt.figure(figsize=(12, 8))
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        
        plt.scatter(model_data['training_time'], model_data['accuracy'], 
                   label=model, alpha=0.7)
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Training Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/task3/plots/accuracy_vs_training_time.png')
    plt.close()