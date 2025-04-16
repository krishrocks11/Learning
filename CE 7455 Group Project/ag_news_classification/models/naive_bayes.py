import logging
import numpy as np
import time
import random
import yaml
import os
import torch
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

class NaiveBayesClassifier:
    """Naive Bayes classifier for text classification"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
        # Select vectorizer based on config
        if config['vectorizer'] == 'count':
            vectorizer = CountVectorizer(
                max_features=config.get('max_features', None),
                stop_words=config.get('stop_words', 'english'),
                ngram_range=tuple(config.get('ngram_range', (1, 1)))
            )
        elif config['vectorizer'] == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=config.get('max_features', None),
                stop_words=config.get('stop_words', 'english'),
                ngram_range=tuple(config.get('ngram_range', (1, 1)))
            )
        else:
            raise ValueError(f"Unknown vectorizer: {config['vectorizer']}")
        
        # Create pipeline
        self.model = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', MultinomialNB(alpha=config.get('alpha', 1.0)))
        ])
        
        logging.info(f"Initialized Naive Bayes classifier with {config['vectorizer']} vectorizer")
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """
        Train the Naive Bayes classifier
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels
            val_texts: List of validation texts
            val_labels: List of validation labels
            
        Returns:
            history: Training history
        """
        logging.info("Starting Naive Bayes model training")
        
        history = {
            'training_time': 0,
            'val_accuracy': 0
        }
        
        # Train model
        start_time = time.time()
        self.model.fit(train_texts, train_labels)
        training_time = time.time() - start_time
        history['training_time'] = training_time
        
        logging.info(f"Training completed in {training_time:.2f} seconds")
        
        # Validate if validation data is provided
        if val_texts is not None and val_labels is not None:
            val_predictions = self.model.predict(val_texts)
            val_accuracy = accuracy_score(val_labels, val_predictions)
            history['val_accuracy'] = val_accuracy
            logging.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        return history
    
    def evaluate(self, test_texts, test_labels):
        """
        Evaluate the Naive Bayes classifier
        
        Args:
            test_texts: List of test texts
            test_labels: List of test labels
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        logging.info("Evaluating Naive Bayes model")
        
        # Measure inference time
        start_time = time.time()
        predictions = self.model.predict(test_texts)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'inference_time': inference_time,
            'avg_inference_time': inference_time / len(test_texts),
            'classification_report': report,
            'predictions': predictions,
            'labels': test_labels
        }
        
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Total inference time: {inference_time:.4f} seconds")
        logging.info(f"Average inference time: {metrics['avg_inference_time']:.6f} seconds per sample")
        
        return metrics
    
    def predict(self, texts):
        """
        Make predictions on new texts
        
        Args:
            texts: List of texts to predict
            
        Returns:
            predictions: List of predicted labels
        """
        return self.model.predict(texts)

def optimize_hyperparameters(train_texts, train_labels, val_texts, val_labels, param_grid, num_trials=10):
    """
    Perform hyperparameter optimization for Naive Bayes
    
    Args:
        train_texts: List of training texts
        train_labels: List of training labels
        val_texts: List of validation texts
        val_labels: List of validation labels
        param_grid: Dictionary of parameters to try
        num_trials: Number of random trials
        
    Returns:
        best_params: Best hyperparameters
        best_model: Best model
        best_val_accuracy: Best validation accuracy
    """
    logging.info("Starting hyperparameter optimization for Naive Bayes")
    
    best_val_accuracy = 0.0
    best_params = None
    best_model = None
    
    # Random search
    for trial in range(num_trials):
        # Randomly sample parameters
        current_params = {}
        for param, values in param_grid.items():
            if isinstance(values, list):
                current_params[param] = random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], (int, float)):
                # For continuous parameters (min, max)
                if isinstance(values[0], float):
                    current_params[param] = random.uniform(values[0], values[1])
                else:
                    current_params[param] = random.randint(values[0], values[1])
            elif param == 'ngram_range' and isinstance(values, list) and len(values) == 2:
                # Special handling for ngram_range which should be a tuple
                current_params[param] = tuple(values)
            else:
                current_params[param] = values
        
        logging.info(f"Trial {trial+1}/{num_trials}: {current_params}")
        
        # Initialize and train model
        model = NaiveBayesClassifier(current_params)
        history = model.train(train_texts, train_labels, val_texts, val_labels)
        
        # Get validation accuracy
        val_accuracy = history.get('val_accuracy', 0)
        
        logging.info(f"Trial {trial+1}/{num_trials}: Validation accuracy: {val_accuracy:.4f}")
        
        # Check if this is the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = current_params
            best_model = model
            
            # Save best model configuration
            with open('config/naive_bayes_config_optimized.yaml', 'w') as f:
                best_params_yaml = {k: list(v) if isinstance(v, tuple) else v for k, v in best_params.items()}
                yaml.dump(best_params_yaml, f)
            
            # Save the best model
            os.makedirs('models/saved', exist_ok=True)
            with open(f'models/saved/naive_bayes_optimized_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)
    
    logging.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logging.info(f"Best parameters: {best_params}")
    
    return best_params, best_model, best_val_accuracy