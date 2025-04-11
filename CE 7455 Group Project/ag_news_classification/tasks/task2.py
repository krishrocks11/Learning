import logging
import os
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import numpy as np
from tqdm import tqdm

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
    Run Task 2: Impact of Data Augmentation on Topic Classification
    
    Args:
        config: Configuration dictionary
        model_type: Specific model to run (if None, run all models)
    """
    logging.info("Starting Task 2: Impact of Data Augmentation on Topic Classification")
    
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Get device (CPU or GPU)
    device = get_device()
    
    # Load AG News dataset
    data = load_ag_news()
    
    # Define augmentation techniques
    augmentation_techniques = ['synonym', 'backtranslation', 'deletion', 'swap']
    
    # Create results directory
    os.makedirs('results/task2', exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'model': [],
        'augmentation': [],
        'accuracy': [],
        'training_time': [],
        'inference_time': []
    }
    
    # Run baseline models (no augmentation)
    baseline_metrics = {}
    
    if model_type is None or model_type == 'bert':
        with timer("BERT baseline model"):
            baseline_metrics['bert'] = run_bert_model(data, device, config, None)
            results['model'].append('bert')
            results['augmentation'].append('none')
            results['accuracy'].append(baseline_metrics['bert']['accuracy'])
            results['training_time'].append(baseline_metrics['bert']['training_time'])
            results['inference_time'].append(baseline_metrics['bert']['avg_inference_time'])
    
    if model_type is None or model_type == 'naive_bayes':
        with timer("Naive Bayes baseline model"):
            baseline_metrics['naive_bayes'] = run_naive_bayes_model(data, config, None)
            results['model'].append('naive_bayes')
            results['augmentation'].append('none')
            results['accuracy'].append(baseline_metrics['naive_bayes']['accuracy'])
            results['training_time'].append(baseline_metrics['naive_bayes']['training_time'])
            results['inference_time'].append(baseline_metrics['naive_bayes']['avg_inference_time'])
    
    if model_type is None or model_type == 'neural_net':
        with timer("Neural Network baseline model"):
            baseline_metrics['neural_net'] = run_neural_net_model(data, device, config, None)
            results['model'].append('neural_net')
            results['augmentation'].append('none')
            results['accuracy'].append(baseline_metrics['neural_net']['accuracy'])
            results['training_time'].append(baseline_metrics['neural_net']['training_time'])
            results['inference_time'].append(baseline_metrics['neural_net']['avg_inference_time'])
    
    # Run models with different augmentation techniques
    for technique in augmentation_techniques:
        logging.info(f"Running models with {technique} augmentation")
        
        # Apply augmentation to training data
        augmented_data = augment_data(data, [technique])
        
        if model_type is None or model_type == 'bert':
            with timer(f"BERT model with {technique} augmentation"):
                metrics = run_bert_model(augmented_data, device, config, technique)
                results['model'].append('bert')
                results['augmentation'].append(technique)
                results['accuracy'].append(metrics['accuracy'])
                results['training_time'].append(metrics['training_time'])
                results['inference_time'].append(metrics['avg_inference_time'])
        
        if model_type is None or model_type == 'naive_bayes':
            with timer(f"Naive Bayes model with {technique} augmentation"):
                metrics = run_naive_bayes_model(augmented_data, config, technique)
                results['model'].append('naive_bayes')
                results['augmentation'].append(technique)
                results['accuracy'].append(metrics['accuracy'])
                results['training_time'].append(metrics['training_time'])
                results['inference_time'].append(metrics['avg_inference_time'])
        
        if model_type is None or model_type == 'neural_net':
            with timer(f"Neural Network model with {technique} augmentation"):
                metrics = run_neural_net_model(augmented_data, device, config, technique)
                results['model'].append('neural_net')
                results['augmentation'].append(technique)
                results['accuracy'].append(metrics['accuracy'])
                results['training_time'].append(metrics['training_time'])
                results['inference_time'].append(metrics['avg_inference_time'])
    
    # Create and save results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/task2/augmentation_results.csv', index=False)
    
    # Plot results
    plot_augmentation_results(results_df)
    
    logging.info("Task 2 completed successfully")
    
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

def run_bert_model(data, device, config, augmentation_name):
    """Run pre-trained AG News-specific BERT model with/without augmentation"""
    prefix = 'baseline' if augmentation_name is None else augmentation_name
    
    logging.info(f"Using pre-trained AG News-specific BERT model ({prefix})")
    
    bert_config = {
        'pretrained_model': 'lucasresck/bert-base-cased-ag-news',
        'num_classes': 4,
        'dropout_rate': 0.1,
        'save_model': True
    }
    
    # Update model save path for this specific augmentation
    bert_config['model_save_path'] = f'models/saved/bert_{prefix}_model.pt'
    
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
    metrics['training_time'] = training_time  # Actually just inference time since no training
    
    # Save pseudo-training history
    pd.DataFrame(history).to_csv(f'results/task2/bert_{prefix}_history.csv', index=False)
    
    return metrics

def run_naive_bayes_model(data, config, augmentation_name):
    """Run Naive Bayes model with/without augmentation"""
    prefix = 'baseline' if augmentation_name is None else augmentation_name
    
    logging.info(f"Running Naive Bayes model ({prefix})")
    
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
    metrics['training_time'] = history['training_time']
    
    # Save metrics
    pd.DataFrame({k: [v] for k, v in history.items()}).to_csv(
        f'results/task2/naive_bayes_{prefix}_history.csv', index=False
    )
    
    return metrics

def run_neural_net_model(data, device, config, augmentation_name):
    """Run Neural Network model with/without augmentation"""
    prefix = 'baseline' if augmentation_name is None else augmentation_name
    
    logging.info(f"Running Neural Network model ({prefix})")
    
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
    
    # Update model save path for this specific augmentation
    nn_config['model_save_path'] = f'models/saved/{nn_config["model_type"]}_{prefix}_model.pt'
    
    # Ensure required parameters are present with defaults if not found
    if 'num_classes' not in nn_config:
        nn_config['num_classes'] = 4  # AG News has 4 classes

    # Ensure min_freq is present
    if 'min_freq' not in nn_config:
        nn_config['min_freq'] = 2
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
    
    # Save training history
    pd.DataFrame(history).to_csv(f'results/task2/{nn_config["model_type"]}_{prefix}_history.csv', index=False)
    
    return metrics

def plot_augmentation_results(results_df):
    """
    Plot augmentation results
    
    Args:
        results_df: DataFrame with results
    """
    # Create accuracy plot
    plt.figure(figsize=(12, 8))
    
    # Reshape data for seaborn
    pivot_acc = results_df.pivot(index='model', columns='augmentation', values='accuracy')
    
    # Plot heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(pivot_acc, annot=True, cmap='Blues', fmt='.4f')
    plt.title('Accuracy by Model and Augmentation')
    
    # Plot bar chart
    plt.subplot(1, 2, 2)
    sns.barplot(x='augmentation', y='accuracy', hue='model', data=results_df)
    plt.title('Accuracy by Augmentation Technique')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/task2/augmentation_accuracy.png')
    plt.close()
    
    # Create training time plot
    plt.figure(figsize=(12, 8))
    
    # Plot training time
    plt.subplot(1, 2, 1)
    sns.barplot(x='model', y='training_time', hue='augmentation', data=results_df)
    plt.title('Training Time by Model and Augmentation')
    plt.ylabel('Training Time (seconds)')
    
    # Plot inference time
    plt.subplot(1, 2, 2)
    sns.barplot(x='model', y='inference_time', hue='augmentation', data=results_df)
    plt.title('Inference Time by Model and Augmentation')
    plt.ylabel('Inference Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('results/task2/augmentation_time.png')
    plt.close()