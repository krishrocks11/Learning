import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np
import random
import yaml
import os

class TextCNN(nn.Module):
    """Convolutional Neural Network for text classification"""
    
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout_rate=0.5, padding_idx=0):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Fully connected layer
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        logging.info(f"Initialized CNN model with vocab_size={vocab_size}, "
                     f"embedding_dim={embedding_dim}, filters={num_filters}, "
                     f"filter_sizes={filter_sizes}")
    
    def forward(self, x):
        """
        Forward pass of the CNN
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            
        Returns:
            logits: Classification logits
        """
        # Embedding Layer [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)
        
        # Permute for convolution [batch_size, embedding_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolutions and max-over-time pooling
        conv_results = []
        for conv in self.convs:
            # Convolution [batch_size, num_filters, seq_len - filter_size + 1]
            conved = F.relu(conv(embedded))
            
            # Max pooling [batch_size, num_filters, 1]
            pooled = F.max_pool1d(conved, conved.shape[2])
            
            # Add to results [batch_size, num_filters]
            conv_results.append(pooled.squeeze(2))
        
        # Concatenate results from different filter sizes [batch_size, num_filters * len(filter_sizes)]
        cat = torch.cat(conv_results, dim=1)
        
        # Apply dropout
        dropped = self.dropout(cat)
        
        # Fully connected layer [batch_size, num_classes]
        logits = self.fc(dropped)
        
        return logits

class TextRNN(nn.Module):
    """Recurrent Neural Network for text classification"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1, 
                 bidirectional=True, dropout_rate=0.5, padding_idx=0, cell_type='lstm'):
        super(TextRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # RNN layer
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        if cell_type.lower() == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, 
                               hidden_dim, 
                               num_layers=num_layers, 
                               bidirectional=bidirectional, 
                               batch_first=True,
                               dropout=dropout_rate if num_layers > 1 else 0)
        elif cell_type.lower() == 'gru':
            self.rnn = nn.GRU(embedding_dim, 
                              hidden_dim, 
                              num_layers=num_layers, 
                              bidirectional=bidirectional, 
                              batch_first=True,
                              dropout=dropout_rate if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unsupported RNN cell type: {cell_type}")
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        logging.info(f"Initialized {cell_type.upper()} model with vocab_size={vocab_size}, "
                     f"embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, "
                     f"bidirectional={bidirectional}")
    
    def forward(self, x):
        """
        Forward pass of the RNN
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            
        Returns:
            logits: Classification logits
        """
        # Embedding Layer [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)
        
        # RNN Layer [batch_size, seq_len, hidden_dim * num_directions]
        output, _ = self.rnn(embedded)
        
        # Get the output from the last time step
        # [batch_size, hidden_dim * num_directions]
        last_hidden = output[:, -1, :]
        
        # Apply dropout
        dropped = self.dropout(last_hidden)
        
        # Fully connected layer [batch_size, num_classes]
        logits = self.fc(dropped)
        
        return logits

def build_vocab(train_texts, min_freq=1):
    """
    Build vocabulary from training texts without using torchtext
    
    Args:
        train_texts: List of training texts
        min_freq: Minimum frequency for a token to be included
        
    Returns:
        vocab: Vocabulary object (custom implementation)
        word_to_idx: Word to index mapping
    """
    # Count word frequencies
    word_counts = {}
    for text in train_texts:
        for word in text.split():
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    
    # Filter by minimum frequency
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Create vocabulary mapping
    word_to_idx = {'<pad>': 0, '<unk>': 1}
    for i, word in enumerate(filtered_words, start=2):
        word_to_idx[word] = i
    
    # Create a simple vocab class with necessary methods
    class SimpleVocab:
        def __init__(self, word_to_idx):
            self.word_to_idx = word_to_idx
            self.idx_to_word = {idx: word for word, idx in word_to_idx.items()}
            self.default_index = 1  # <unk> index
        
        def __len__(self):
            return len(self.word_to_idx)
        
        def __getitem__(self, word):
            return self.word_to_idx.get(word, self.default_index)
        
        def get_itos(self):
            return [self.idx_to_word[i] for i in range(len(self))]
        
        def set_default_index(self, index):
            self.default_index = index
    
    vocab = SimpleVocab(word_to_idx)
    
    logging.info(f"Vocabulary built with {len(vocab)} tokens")
    
    return vocab, word_to_idx

def text_to_tensor(text, word_to_idx, max_length=100):
    """
    Convert text to tensor of token indices
    
    Args:
        text: Text to convert
        word_to_idx: Word to index mapping
        max_length: Maximum sequence length
        
    Returns:
        tensor: Tensor of token indices
    """
    tokens = text.split()
    indices = [word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens]
    
    # Truncate or pad to max_length
    if len(indices) > max_length:
        indices = indices[:max_length]
    else:
        indices = indices + [word_to_idx['<pad>']] * (max_length - len(indices))
    
    return torch.tensor(indices, dtype=torch.long)

class NeuralNetDataset(torch.utils.data.Dataset):
    """Dataset for neural network models"""
    
    def __init__(self, texts, labels, word_to_idx, max_length=100):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        tensor = text_to_tensor(text, self.word_to_idx, self.max_length)
        
        return {
            'text': tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_neural_net_dataloaders(data, word_to_idx, batch_size=32, max_length=100):
    """
    Create DataLoaders for neural network models
    
    Args:
        data: Dictionary with train, val, test splits
        word_to_idx: Word to index mapping
        batch_size: Batch size
        max_length: Maximum sequence length
        
    Returns:
        dataloaders: Dictionary with train, val, test DataLoaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        texts, labels = data[split]
        dataset = NeuralNetDataset(texts, labels, word_to_idx, max_length)
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=2
        )
    
    return dataloaders

def train_neural_net(model, dataloaders, device, config):
    """
    Train the neural network model with early stopping
    
    Args:
        model: Neural network model
        dataloaders: Dictionary with train, val, test DataLoaders
        device: Device to use (CPU or GPU)
        config: Training configuration
        
    Returns:
        model: Trained model
        history: Training history
    """
    logging.info(f"Starting {config.get('model_type', 'Neural Network')} model training")
    
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize tracking variables
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'time_per_epoch': []
    }
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    best_model_state = None
    patience = config.get('patience', 3)  # Get patience from config or default to 3
    patience_counter = 0
    
    # Training loop
    for epoch in range(config['num_epochs']):
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(dataloaders['train'], desc="Training"):
            inputs = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
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
                inputs = batch['text'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, labels)
                
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
            best_model_state = model.state_dict().copy()
            if config.get('save_model', False):
                torch.save(model.state_dict(), config['model_save_path'])
                logging.info(f"Model saved to {config['model_save_path']}")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
            logging.info(f"Validation accuracy did not improve. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    logging.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    return model, history

def evaluate_neural_net(model, dataloader, device):
    """
    Evaluate the neural network model
    
    Args:
        model: Neural network model
        dataloader: Test dataloader
        device: Device to use (CPU or GPU)
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    logging.info("Evaluating neural network model")
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
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

def optimize_hyperparameters(data, device, param_grid, num_trials=5):
    """
    Perform hyperparameter optimization for neural network models
    
    Args:
        data: Data dictionary with train, val and test splits
        device: Device to use
        param_grid: Dictionary of parameters to try
        num_trials: Number of random trials
        
    Returns:
        best_params: Best hyperparameters
        best_model: Best model
        best_val_accuracy: Best validation accuracy
    """
    logging.info("Starting hyperparameter optimization for Neural Network")
    
    best_val_accuracy = 0.0
    best_params = None
    best_model = None
    best_history = None
    
    # Build vocabulary using all trials with the same vocab
    train_texts, train_labels = data['train']
    min_freq = param_grid.get('min_freq', [2])[0] if isinstance(param_grid.get('min_freq', 2), list) else param_grid.get('min_freq', 2)
    vocab, word_to_idx = build_vocab(train_texts, min_freq)
    
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
            elif param == 'filter_sizes' and isinstance(values, list):
                # Special handling for filter_sizes
                if all(isinstance(x, int) for x in values):
                    # If all are integers, sample 2-3 random sizes
                    num_filters = random.randint(2, min(3, len(values)))
                    current_params[param] = sorted(random.sample(values, num_filters))
                else:
                    # Otherwise just pick one of the filter size combinations
                    current_params[param] = random.choice(values)
            else:
                current_params[param] = values
                
        # Always use the vocabulary size from the built vocab
        current_params['vocab_size'] = len(vocab)
        
        logging.info(f"Trial {trial+1}/{num_trials}: {current_params}")
        
        # Create config for training
        model_type = current_params.get('model_type', 'cnn')
        
        config = {
            'model_type': model_type,
            'vocab_size': current_params['vocab_size'],
            'embedding_dim': current_params.get('embedding_dim', 300),
            'max_length': current_params.get('max_length', 100),
            'batch_size': current_params.get('batch_size', 64),
            'learning_rate': current_params.get('learning_rate', 0.001),
            'num_epochs': current_params.get('num_epochs', 5),
            'dropout_rate': current_params.get('dropout_rate', 0.5),
            'model_save_path': f'models/saved/{model_type}_trial{trial}.pt',
            'save_model': False
        }
        
        # Add model-specific parameters
        if model_type.lower() == 'cnn':
            config.update({
                'num_filters': current_params.get('num_filters', 100),
                'filter_sizes': current_params.get('filter_sizes', [3, 4, 5])
            })
        else:  # RNN, LSTM, GRU
            config.update({
                'hidden_dim': current_params.get('hidden_dim', 256),
                'num_layers': current_params.get('num_layers', 2),
                'bidirectional': current_params.get('bidirectional', True),
                'cell_type': current_params.get('cell_type', 'lstm')
            })
        
        # Create dataloaders
        batch_size = config['batch_size']
        max_length = config['max_length']
        dataloaders = create_neural_net_dataloaders(data, word_to_idx, batch_size, max_length)
        
        # Initialize model
        if model_type.lower() == 'cnn':
            model = TextCNN(
                vocab_size=config['vocab_size'],
                embedding_dim=config['embedding_dim'],
                num_filters=config['num_filters'],
                filter_sizes=config['filter_sizes'],
                num_classes=4,
                dropout_rate=config['dropout_rate']
            )
        else:  # RNN, LSTM, GRU
            model = TextRNN(
                vocab_size=config['vocab_size'],
                embedding_dim=config['embedding_dim'],
                hidden_dim=config['hidden_dim'],
                num_classes=4,
                num_layers=config['num_layers'],
                bidirectional=config['bidirectional'],
                dropout_rate=config['dropout_rate'],
                cell_type=config['cell_type']
            )
        
        # Train model
        trained_model, history = train_neural_net(model, dataloaders, device, config)
        
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
            with open(f'config/neural_net_{model_type}_config_optimized.yaml', 'w') as f:
                yaml.dump(current_params, f)
    
    # Save the best model
    if best_model is not None:
        os.makedirs('models/saved', exist_ok=True)
        torch.save(best_model.state_dict(), f'models/saved/{model_type}_optimized_model.pt')
    
    logging.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logging.info(f"Best parameters: {best_params}")
    
    return best_params, best_model, best_val_accuracy