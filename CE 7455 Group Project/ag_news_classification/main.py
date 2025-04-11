import argparse
import logging
import os
import sys
import torch
import yaml
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Add NLTK import
import nltk

from tasks import task1, task2, task3
from src.utils import setup_logging

def download_nltk_resources():
    """Download required NLTK resources if they aren't already available"""
    logging.info("Checking and downloading required NLTK resources")
    
    # List of NLTK resources needed for the project
    resources = [
        'punkt',
        'wordnet',
        'omw-1.4',  # Open Multilingual WordNet
        'averaged_perceptron_tagger',
        'stopwords'
    ]
    
    for resource in resources:
        try:
            logging.info(f"Checking NLTK resource: {resource}")
            nltk.data.find(f'tokenizers/{resource}')
            logging.info(f"NLTK resource {resource} already downloaded")
        except LookupError:
            logging.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)
            logging.info(f"Successfully downloaded {resource}")

def parse_args():
    parser = argparse.ArgumentParser(description='AG News Classification')
    parser.add_argument('--task', type=int, choices=[1, 2, 3], help='Task to run (1, 2, or 3)')
    parser.add_argument('--all', action='store_true', help='Run all tasks')
    parser.add_argument('--config', type=str, default='config/train_config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, choices=['bert', 'naive_bayes', 'neural_net'], 
                        help='Model to use (only if running a specific task)')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    return parser.parse_args()

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        sys.exit(1)

def cleanup_memory():
    """Clean up memory after task execution, particularly important for GPU usage"""
    # Run Python's garbage collector
    gc.collect()
    
    # If using GPU, empty CUDA cache
    if torch.cuda.is_available():
        with torch.no_grad():
            torch.cuda.empty_cache()
        logging.info("CUDA memory cache cleared")
    
    logging.info("Memory cleanup performed")

def compile_results(results_dir='results'):
    """Compile results from all tasks into a summary report"""
    logging.info("Compiling results summary across all tasks")
    
    # Create summary directory
    summary_dir = os.path.join(results_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create summary dataframe
    summary = {
        'task': [],
        'model': [],
        'configuration': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'training_time': [],
        'inference_time': []
    }
    
    # Aggregate results from Task 1
    try:
        task1_results = pd.read_csv(os.path.join(results_dir, 'task1', 'model_comparison.csv'))
        for _, row in task1_results.iterrows():
            summary['task'].append('Task 1 - Basic Classification')
            summary['model'].append(row['Model'])
            summary['configuration'].append('Default')
            summary['accuracy'].append(row['Accuracy'])
            # Add other metrics if available in your results files
            summary['precision'].append(None)
            summary['recall'].append(None)
            summary['f1'].append(None)
            summary['training_time'].append(None)
            summary['inference_time'].append(row['Inference Time (s)'])
    except Exception as e:
        logging.warning(f"Could not load Task 1 results: {e}")
    
    # Aggregate results from Task 2
    try:
        task2_results = pd.read_csv(os.path.join(results_dir, 'task2', 'augmentation_results.csv'))
        for _, row in task2_results.iterrows():
            summary['task'].append('Task 2 - Data Augmentation')
            summary['model'].append(row['model'])
            summary['configuration'].append(f"Aug: {row['augmentation']}")
            summary['accuracy'].append(row['accuracy'])
            # Add other metrics if available
            summary['precision'].append(None)
            summary['recall'].append(None)
            summary['f1'].append(None)
            summary['training_time'].append(row['training_time'])
            summary['inference_time'].append(row['inference_time'])
    except Exception as e:
        logging.warning(f"Could not load Task 2 results: {e}")
    
    # Aggregate results from Task 3
    try:
        task3_results = pd.read_csv(os.path.join(results_dir, 'task3', 'dataset_size_augmentation_results.csv'))
        for _, row in task3_results.iterrows():
            summary['task'].append('Task 3 - Dataset Size & Augmentation')
            summary['model'].append(row['model'])
            summary['configuration'].append(f"Size: {row['dataset_size']*100}%, Aug: {row['augmentation']}")
            summary['accuracy'].append(row['accuracy'])
            summary['precision'].append(row['precision'])
            summary['recall'].append(row['recall'])
            summary['f1'].append(row['f1'])
            summary['training_time'].append(row['training_time'])
            summary['inference_time'].append(row['inference_time'])
    except Exception as e:
        logging.warning(f"Could not load Task 3 results: {e}")
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary)
    
    # Save to CSV
    summary_df.to_csv(os.path.join(summary_dir, 'all_tasks_summary.csv'), index=False)
    
    # Generate summary visualizations
    try:
        # Accuracy comparison across tasks and models
        plt.figure(figsize=(12, 8))
        sns.barplot(x='model', y='accuracy', hue='task', data=summary_df)
        plt.title('Accuracy Comparison Across Tasks and Models')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'accuracy_comparison.png'))
        plt.close()
        
        # Training time comparison
        valid_time_data = summary_df.dropna(subset=['training_time'])
        if len(valid_time_data) > 0:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='model', y='training_time', hue='task', data=valid_time_data)
            plt.title('Training Time Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, 'training_time_comparison.png'))
            plt.close()
        
        logging.info(f"Summary report saved to {summary_dir}")
    except Exception as e:
        logging.error(f"Error creating summary visualizations: {e}")
    
    return summary_df

def main():
    args = parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/ag_news_classification_{timestamp}.log'
    setup_logging(log_file, level=args.log_level)
    
    logging.info("Starting AG News Classification Project")
    
    # Download required NLTK resources
    download_nltk_resources()
    
    # Load configuration
    config = load_config(args.config)
    logging.info(f"Loaded configuration from {args.config}")
    
    try:
        if args.all:
            logging.info("Running all tasks sequentially")
            for task_num, task_fn in tqdm(enumerate([task1.run, task2.run, task3.run], 1), 
                                       total=3, desc="Tasks Progress", position=0):
                logging.info(f"Running task {task_num}/3")
                task_fn(config)
                cleanup_memory()
            compile_results()
        elif args.task:
            logging.info(f"Running task {args.task}")
            if args.task == 1:
                task1.run(config, model_type=args.model, optimize=args.optimize)
                cleanup_memory()
            elif args.task == 2:
                task2.run(config, model_type=args.model)
                cleanup_memory()
            elif args.task == 3:
                task3.run(config, model_type=args.model)
                cleanup_memory()
        else:
            logging.error("Please specify a task to run (--task) or use --all to run all tasks")
            sys.exit(1)
            
        logging.info("All tasks completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()