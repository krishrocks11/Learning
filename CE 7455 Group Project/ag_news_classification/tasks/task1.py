import logging
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import yaml

from src.data_loader import load_ag_news, create_data_loaders
from src.utils import get_device, set_seed, timer
from models.bert_classifier import BertClassifier, get_bert_tokenizer, train_bert, evaluate_bert, optimize_hyperparameters as optimize_bert
from models.bert_classifier import use_pretrained_bert
from models.naive_bayes import NaiveBayesClassifier, optimize_hyperparameters as optimize_naive_bayes
from models.neural_net import (TextCNN, TextRNN, build_vocab, create_neural_net_dataloaders,
                              train_neural_net, evaluate_neural_net, optimize_hyperparameters as optimize_neural_net)

def run(config, model_type=None, optimize=False):
    """
    Run Task 1: Topic Classification on AG News Using Three Models
    
    Args:
        config: Configuration dictionary
        model_type: Specific model to run (if None, run all models)
        optimize: Whether to perform hyperparameter optimization
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    logging.info("Starting Task 1: Topic Classification on AG News")
    
    # Check if optimized configs already exist when optimize=True
    if optimize:
        optimized_configs_exist = False
        
        # Check for optimized BERT config
        if os.path.exists('config/bert_config_optimized.yaml'):
            optimized_configs_exist = True
        
        # Check for optimized Naive Bayes config
        if os.path.exists('config/naive_bayes_config_optimized.yaml'):
            optimized_configs_exist = True
        
        # Check for optimized neural network configs (CNN or RNN)
        if os.path.exists('config/neural_net_cnn_config_optimized.yaml') or \
           os.path.exists('config/neural_net_rnn_config_optimized.yaml'):
            optimized_configs_exist = True
        
        if optimized_configs_exist:
            # In command line mode, we can't get user input - provide warning
            logging.warning("*" * 80)
            logging.warning("ATTENTION: Optimized configurations already exist!")
            logging.warning("Running with --optimize will overwrite these configurations.")
            logging.warning("If you want to keep existing optimized configs, press Ctrl+C now.")
            logging.warning("Continuing optimization in 5 seconds...")
            logging.warning("*" * 80)
            
            # Wait 5 seconds to give the user a chance to abort
            import time
            for i in range(5, 0, -1):
                logging.warning(f"Starting optimization in {i} seconds...")
                time.sleep(1)
    
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Get device (CPU or GPU)
    device = get_device()
    
    # Load AG News dataset
    data = load_ag_news()
    
    # Create results directory
    os.makedirs('results/task1', exist_ok=True)
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Run models based on model_type
    if model_type is None or model_type == 'bert':
        with timer("BERT model"):
            if optimize:
                metrics['bert'] = run_bert_model_with_optimization(data, device, config)
            else:
                metrics['bert'] = run_bert_model(data, device, config)
    
    if model_type is None or model_type == 'naive_bayes':
        with timer("Naive Bayes model"):
            if optimize:
                metrics['naive_bayes'] = run_naive_bayes_model_with_optimization(data, config)
            else:
                metrics['naive_bayes'] = run_naive_bayes_model(data, config)
    
    if model_type is None or model_type == 'neural_net':
        with timer("Neural Network model"):
            if optimize:
                metrics['neural_net'] = run_neural_net_model_with_optimization(data, device, config)
            else:
                metrics['neural_net'] = run_neural_net_model(data, device, config)
    
    # Compare models
    if len(metrics) > 1:
        compare_models(metrics)
    
    logging.info("Task 1 completed successfully")
    
    return metrics

def run_bert_model_with_optimization(data, device, config):
    """
    Skip optimization for BERT since we're using the pre-trained model as is
    """
    logging.info("Using pre-trained BERT model (optimization skipped)")
    return run_bert_model(data, device, config)

def run_naive_bayes_model_with_optimization(data, config):
    """
    Run Naive Bayes model with hyperparameter optimization
    
    Args:
        data: AG News dataset
        config: Configuration dictionary
        
    Returns:
        metrics: Evaluation metrics
    """
    logging.info("Running Naive Bayes model with hyperparameter optimization")
    
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['val']
    test_texts, test_labels = data['test']
    
    # Define parameter grid for optimization
    param_grid = {
        'vectorizer': ['count', 'tfidf'],
        'max_features': [5000, 10000, 20000, None],
        'ngram_range': [(1, 1), (1, 2), (1, 3)],
        'alpha': [0.1, 0.5, 1.0, 2.0]
    }
    
    # Perform hyperparameter optimization
    best_params, best_model, _ = optimize_naive_bayes(
        train_texts, train_labels, val_texts, val_labels, param_grid, num_trials=10
    )
    
    # Evaluate best model
    metrics = best_model.evaluate(test_texts, test_labels)
    
    # Create confusion matrix
    plot_confusion_matrix(
        metrics['predictions'], 
        metrics['labels'], 
        'results/task1/naive_bayes_optimized_confusion_matrix.png',
        title='Naive Bayes (Optimized) Confusion Matrix'
    )
    
    return metrics

def run_neural_net_model_with_optimization(data, device, config):
    """
    Run Neural Network model with hyperparameter optimization
    
    Args:
        data: AG News dataset
        device: Device to use (CPU or GPU)
        config: Configuration dictionary
        
    Returns:
        metrics: Evaluation metrics
    """
    logging.info("Running Neural Network model with hyperparameter optimization")
    
    # Define parameter grid for optimization
    param_grid = {
        'model_type': ['cnn', 'rnn'],
        # Common parameters
        'embedding_dim': [100, 200, 300],
        'max_length': [100, 150, 200],
        'batch_size': [32, 64, 128],
        'learning_rate': [0.0005, 0.001, 0.002],
        'dropout_rate': [0.3, 0.5, 0.7],
        'num_epochs': [5, 10],
        # CNN-specific parameters
        'num_filters': [50, 100, 200],
        'filter_sizes': [[2, 3, 4], [3, 4, 5], [2, 4, 6]],
        # RNN-specific parameters
        'hidden_dim': [128, 256, 512],
        'num_layers': [1, 2],
        'bidirectional': [True, False],
        'cell_type': ['lstm', 'gru']
    }
    
    # Perform hyperparameter optimization
    best_params, best_model, _ = optimize_neural_net(data, device, param_grid, num_trials=5)
    
    # Rebuild vocabulary and create dataloaders with best parameters
    train_texts, _ = data['train']
    vocab, word_to_idx = build_vocab(train_texts, best_params.get('min_freq', 2))
    
    dataloaders = create_neural_net_dataloaders(
        data, word_to_idx, best_params.get('batch_size', 64), best_params.get('max_length', 100)
    )
    
    # Evaluate best model
    metrics = evaluate_neural_net(best_model, dataloaders['test'], device)
    
    # Create confusion matrix
    model_type = best_params.get('model_type', 'cnn')
    plot_confusion_matrix(
        metrics['predictions'], 
        metrics['labels'], 
        f'results/task1/{model_type}_optimized_confusion_matrix.png',
        title=f'{model_type.upper()} (Optimized) Confusion Matrix'
    )
    
    return metrics

def run_bert_model(data, device, config):
    """Use pre-trained AG News-specific BERT model for topic classification"""
    logging.info("Using pre-trained AG News-specific BERT model for topic classification")
    
    bert_config = {
        'pretrained_model': 'lucasresck/bert-base-cased-ag-news',
        'num_classes': 4,
        'dropout_rate': 0.1,
        'model_save_path': 'models/saved/ag_news_bert_model.pt',
        'save_model': True
    }
    
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
    trained_model, history = use_pretrained_bert(model, dataloaders, device)
    
    # Evaluate model
    metrics = evaluate_bert(trained_model, dataloaders['test'], device)
    
    # Create confusion matrix
    plot_confusion_matrix(
        metrics['predictions'], 
        metrics['labels'], 
        'results/task1/bert_confusion_matrix.png',
        title='AG News-specific BERT Confusion Matrix'
    )
    
    return metrics

def run_naive_bayes_model(data, config):
    """Run Naive Bayes model for topic classification"""
    logging.info("Running Naive Bayes model for topic classification")
    
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
    
    # Save metrics
    pd.DataFrame({k: [v] for k, v in history.items()}).to_csv('results/task1/naive_bayes_history.csv', index=False)
    
    # Create confusion matrix
    plot_confusion_matrix(
        metrics['predictions'], 
        metrics['labels'], 
        'results/task1/naive_bayes_confusion_matrix.png'
    )
    
    return metrics

def run_neural_net_model(data, device, config):
    """Run Neural Network model for topic classification"""
    logging.info("Running Neural Network model for topic classification")
    
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
                'model_save_path': 'models/saved/neural_net_model.pt',
                'save_model': True
            }
    
    # Ensure required parameters are present with defaults if not found
    if 'num_classes' not in nn_config:
        nn_config['num_classes'] = 4  # AG News has 4 classes
        
    if 'min_freq' not in nn_config:
        nn_config['min_freq'] = 2  # Default value
    
    if 'model_save_path' not in nn_config:
        nn_config['model_save_path'] = 'models/saved/neural_net_model.pt'
        
    if 'save_model' not in nn_config:
        nn_config['save_model'] = True
        
    if 'max_length' not in nn_config:
        nn_config['max_length'] = 100
    
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
            num_classes=nn_config['num_classes'],
            dropout_rate=nn_config['dropout_rate']
        )
    else:  # RNN, LSTM, GRU
        model = TextRNN(
            vocab_size=nn_config['vocab_size'],
            embedding_dim=nn_config['embedding_dim'],
            hidden_dim=nn_config['hidden_dim'],
            num_classes=nn_config['num_classes'],
            num_layers=nn_config['num_layers'],
            bidirectional=nn_config['bidirectional'],
            dropout_rate=nn_config['dropout_rate'],
            cell_type=nn_config['cell_type']
        )
    
    # Train model
    trained_model, history = train_neural_net(model, dataloaders, device, nn_config)
    
    # Evaluate model
    metrics = evaluate_neural_net(trained_model, dataloaders['test'], device)
    
    # Save training history
    pd.DataFrame(history).to_csv(f'results/task1/{nn_config["model_type"]}_history.csv', index=False)
    
    # Plot and save training curves
    plot_training_curves(history, f'results/task1/{nn_config["model_type"]}_training_curves.png')
    
    # Create confusion matrix
    plot_confusion_matrix(
        metrics['predictions'], 
        metrics['labels'], 
        f'results/task1/{nn_config["model_type"]}_confusion_matrix.png'
    )
    
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

def plot_confusion_matrix(predictions, labels, save_path, title='Confusion Matrix'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_models(metrics):
    """Compare and visualize model performance"""
    model_names = list(metrics.keys())
    accuracies = [metrics[model]['accuracy'] for model in model_names]
    inference_times = []
    
    for model in model_names:
        if 'avg_inference_time' in metrics[model]:
            inference_times.append(metrics[model]['avg_inference_time'])
        else:
            inference_times.append(metrics[model].get('inference_time', 0))
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Inference Time (s)': inference_times
    })
    
    # Save comparison to CSV
    comparison.to_csv('results/task1/model_comparison.csv', index=False)
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='Accuracy', data=comparison)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='Inference Time (s)', data=comparison)
    plt.title('Model Inference Time Comparison')
    plt.ylabel('Inference Time (s)')
    
    plt.tight_layout()
    plt.savefig('results/task1/model_comparison.png')
    plt.close()
    
    logging.info(f"Model comparison:\n{comparison}")