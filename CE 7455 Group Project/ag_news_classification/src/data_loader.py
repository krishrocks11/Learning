import logging
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class AGNewsDataset(Dataset):
    """AG News Dataset"""
    
    def __init__(self, texts, labels, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            # For BERT or other transformer models
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # For other models
            return {
                'text': text,
                'label': torch.tensor(label, dtype=torch.long)
            }

def download_ag_news():
    """Download AG News dataset directly without using torchtext"""
    import requests
    import csv
    import io
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    logging.info("Downloading AG News dataset directly")
    
    try:
        os.makedirs('data/raw', exist_ok=True)
        
        # AG News dataset URLs
        train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
        test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
        
        # Check if files already exist
        train_path = 'data/raw/ag_news_train.csv'
        test_path = 'data/raw/ag_news_test.csv'
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            logging.info("AG News files already downloaded, loading from disk")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            return train_df, test_df
        
        # Download and process training data
        logging.info("Downloading AG News training data")
        train_response = requests.get(train_url)
        train_response.raise_for_status()  # Ensure successful download
        
        # Parse CSV content
        train_data = []
        with io.StringIO(train_response.text) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:  # Ensure row has at least label and text
                    label = int(row[0]) - 1  # Convert 1-indexed to 0-indexed
                    text = row[1] + " " + row[2]  # Combine title and description
                    train_data.append({'label': label, 'text': text})
        
        # Download and process test data
        logging.info("Downloading AG News test data")
        test_response = requests.get(test_url)
        test_response.raise_for_status()  # Ensure successful download
        
        # Parse CSV content
        test_data = []
        with io.StringIO(test_response.text) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    label = int(row[0]) - 1  # Convert 1-indexed to 0-indexed
                    text = row[1] + " " + row[2]  # Combine title and description
                    test_data.append({'label': label, 'text': text})
        
        # Convert to pandas DataFrames
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        # Save to CSV
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logging.info(f"AG News dataset downloaded and saved: {len(train_df)} training, {len(test_df)} test samples")
        return train_df, test_df
    
    except Exception as e:
        logging.error(f"Error downloading AG News dataset: {e}")
        raise  # Re-raise the exception to be handled by the caller

def load_ag_news(subset_size=1.0, val_size=0.1):
    """Load AG News dataset from CSV files or download if not present"""
    try:
        train_path = 'data/raw/ag_news_train.csv'
        test_path = 'data/raw/ag_news_test.csv'
        
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            logging.info("Dataset not found. Downloading AG News dataset...")
            train_df, test_df = download_ag_news()
        else:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("AG News dataset loaded from files")
        
        # Apply subset size if needed (for Task 3)
        if subset_size < 1.0:
            train_df = train_df.sample(frac=subset_size, random_state=42)
            logging.info(f"Using {subset_size*100}% of training data: {len(train_df)} samples")
        
        # Split training data into train and validation
        train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42, stratify=train_df['label'])
        
        logging.info(f"Dataset loaded: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test samples")
        
        return {
            'train': (train_df['text'].tolist(), train_df['label'].tolist()),
            'val': (val_df['text'].tolist(), val_df['label'].tolist()),
            'test': (test_df['text'].tolist(), test_df['label'].tolist())
        }
    
    except Exception as e:
        logging.error(f"Error loading AG News dataset: {e}")
        raise # Re-raise the exception to be handled by the caller

def create_data_loaders(data, tokenizer=None, batch_size=32):
    """Create PyTorch DataLoaders for train, validation, and test sets"""
    datasets = {}
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        texts, labels = data[split]
        datasets[split] = AGNewsDataset(texts, labels, tokenizer)
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=2
        )
    
    return dataloaders