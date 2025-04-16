import logging
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.optim import AdamW  # Use PyTorch's implementation instead
from tqdm import tqdm
import time
import numpy as np
import random
import yaml
import os

from src.data_loader import create_data_loaders

class BertClassifier(nn.Module):
    """BERT model for text classification"""
    
    def __init__(self, num_classes=4, pretrained_model='bert-base-uncased', dropout_rate=0.1, use_ag_news_model=False):
        super(BertClassifier, self).__init__()
        
        self.use_ag_news_model = use_ag_news_model
        
        if use_ag_news_model:
            # Use the AG News-specific pre-trained model
            # Fix for warnings - add ignore_mismatched_sizes=True and remove gradient_checkpointing
            self.bert = BertForSequenceClassification.from_pretrained(
                'lucasresck/bert-base-cased-ag-news',
                ignore_mismatched_sizes=True,
                # Force download=False as resume_download is deprecated
                force_download=False
            )
            logging.info(f"Initialized AG News-specific BERT classifier")
        else:
            # Use the standard approach with generic BERT
            # Try different initialization approaches
            try:
                # Try direct initialization first - with updated parameters
                self.bert = BertModel.from_pretrained(
                    pretrained_model, 
                    force_download=False,  # Instead of resume_download
                    ignore_mismatched_sizes=True
                )
            except Exception as e:
                logging.warning(f"Standard initialization failed: {e}")
                
                try:
                    # Try with low_cpu_mem_usage=False
                    self.bert = BertModel.from_pretrained(
                        pretrained_model, 
                        low_cpu_mem_usage=False,
                        force_download=False,
                        ignore_mismatched_sizes=True
                    )
                except Exception as e:
                    logging.warning(f"Second attempt failed: {e}")
                    
                    # Last resort - fallback to config-based initialization
                    from transformers import BertConfig
                    logging.warning("Falling back to manual model construction")
                    config = BertConfig.from_pretrained(pretrained_model)
                    self.bert = BertModel(config)
            
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
            
            logging.info(f"Initialized generic BERT classifier with {pretrained_model}")
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the BERT classifier
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            logits: Classification logits
        """
        if self.use_ag_news_model:
            # AG News-specific model handles classification internally
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits
        else:
            # Standard approach for generic BERT
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

def get_bert_tokenizer(pretrained_model='bert-base-uncased', use_ag_news_model=False):
    """Get BERT tokenizer"""
    if use_ag_news_model:
        return AutoTokenizer.from_pretrained('lucasresck/bert-base-cased-ag-news')
    else:
        return BertTokenizer.from_pretrained(pretrained_model)

def train_bert(model, dataloaders, device, config):
    """
    Train the BERT classifier
    
    Args:
        model: BERT classifier model
        dataloaders: Dictionary of train, val, test dataloaders
        device: Device to use (CPU or GPU)
        config: Training configuration
        
    Returns:
        model: Trained model
        history: Training history
    """
    logging.info("Starting BERT model training")
    
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], eps=config['adam_epsilon'])
    
    # Calculate total steps
    total_steps = len(dataloaders['train']) * config['num_epochs']
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Initialize tracking variables
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'time_per_epoch': []
    }
    
    best_val_accuracy = 0.0
    
    # Training loop
    for epoch in range(config['num_epochs']):
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(dataloaders['train'], desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            # Update weights
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Track loss
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(dataloaders['train'])
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloaders['val'], desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Compute loss
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
                
                # Track loss and accuracy
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / len(dataloaders['val'])
        val_accuracy = val_correct / val_total
        
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Calculate time per epoch
        epoch_time = time.time() - epoch_start_time
        history['time_per_epoch'].append(epoch_time)
        
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']} - "
                     f"Train Loss: {avg_train_loss:.4f}, "
                     f"Val Loss: {avg_val_loss:.4f}, "
                     f"Val Accuracy: {val_accuracy:.4f}, "
                     f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            if config.get('save_model', False):
                torch.save(model.state_dict(), config['model_save_path'])
                logging.info(f"Model saved to {config['model_save_path']}")
    
    logging.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    return model, history

def evaluate_bert(model, dataloader, device):
    """
    Evaluate the BERT classifier
    
    Args:
        model: BERT classifier model
        dataloader: Test dataloader
        device: Device to use (CPU or GPU)
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    logging.info("Evaluating BERT model")
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Get predictions
            _, predictions = torch.max(outputs, 1)
            
            # Convert to lists
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    avg_inference_time = np.mean(inference_times)
    
    metrics = {
        'accuracy': accuracy,
        'avg_inference_time': avg_inference_time,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Average inference time: {avg_inference_time:.6f} seconds per batch")
    
    return metrics

def optimize_hyperparameters(data, device, param_grid, num_trials=3):
    """
    Perform hyperparameter optimization for BERT
    
    Args:
        data: Data dictionary with train, val and test
        device: Device to use
        param_grid: Dictionary of parameters to try
        num_trials: Number of random trials
        
    Returns:
        best_params: Best hyperparameters
        best_model: Best model
        best_val_accuracy: Best validation accuracy
    """
    logging.info("Starting hyperparameter optimization for BERT")
    
    best_val_accuracy = 0.0
    best_params = None
    best_model = None
    best_history = None
    
    # Random search
    for trial in range(num_trials):
        # Randomly sample parameters
        current_params = {}
        for param, values in param_grid.items():
            if isinstance(values, list):
                current_params[param] = random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                # For continuous parameters (min, max)
                if isinstance(values[0], float):
                    current_params[param] = random.uniform(values[0], values[1])
                else:
                    current_params[param] = random.randint(values[0], values[1])
        
        logging.info(f"Trial {trial+1}/{num_trials}: {current_params}")
        
        # Initialize tokenizer and model
        tokenizer = get_bert_tokenizer(current_params.get('pretrained_model', 'bert-base-uncased'))
        dataloaders = create_data_loaders(data, tokenizer, current_params.get('batch_size', 16))
        
        model = BertClassifier(
            num_classes=current_params.get('num_classes', 4),
            pretrained_model=current_params.get('pretrained_model', 'bert-base-uncased'),
            dropout_rate=current_params.get('dropout_rate', 0.1)
        )
        
        # Create config
        config = {
            'learning_rate': current_params.get('learning_rate', 2e-5),
            'adam_epsilon': current_params.get('adam_epsilon', 1e-8),
            'warmup_steps': current_params.get('warmup_steps', 0),
            'max_grad_norm': current_params.get('max_grad_norm', 1.0),
            'num_epochs': current_params.get('num_epochs', 3),
            'batch_size': current_params.get('batch_size', 16),
            'model_save_path': f'models/saved/bert_trial{trial}.pt',
            'save_model': False
        }
        
        # Train model
        trained_model, history = train_bert(model, dataloaders, device, config)
        
        # Get validation accuracy
        val_accuracy = max(history.get('val_accuracy', [0]))
        
        logging.info(f"Trial {trial+1}/{num_trials}: Validation accuracy: {val_accuracy:.4f}")
        
        # Check if this is the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = current_params
            best_model = trained_model
            best_history = history
            
            # Save best model configuration
            with open('config/bert_config_optimized.yaml', 'w') as f:
                yaml.dump(best_params, f)
    
    # Save the best model
    if best_model is not None:
        os.makedirs('models/saved', exist_ok=True)
        torch.save(best_model.state_dict(), f'models/saved/bert_optimized_model.pt')
    
    logging.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logging.info(f"Best parameters: {best_params}")
    
    return best_params, best_model, best_val_accuracy

def use_pretrained_bert(model, dataloaders, device):
    """
    Use pre-trained BERT model without additional training
    
    Args:
        model: BERT classifier model
        dataloaders: Dictionary with train, val, test dataloaders
        device: Device to use (CPU or GPU)
        
    Returns:
        model: Pre-trained model
        history: Empty history (since no training was done)
    """
    logging.info("Using pre-trained BERT model without additional training")
    
    model = model.to(device)
    
    # Create empty history to maintain consistent return structure
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'time_per_epoch': []
    }
    
    # Validate model to get initial metrics
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloaders['val'], desc="Validating pre-trained model"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    # Calculate validation accuracy
    val_accuracy = val_correct / val_total
    history['val_accuracy'].append(val_accuracy)
    
    logging.info(f"Pre-trained BERT model validation accuracy: {val_accuracy:.4f}")
    
    return model, history