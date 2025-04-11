import logging
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logging.warning(f"Error downloading NLTK resources: {e}")

def clean_text(text):
    """
    Clean text by removing special characters, numbers, and extra whitespace
    
    Args:
        text: Text to clean
        
    Returns:
        cleaned_text: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text, lang='english'):
    """
    Remove stopwords from text
    
    Args:
        text: Text to process
        lang: Language for stopwords
        
    Returns:
        text_without_stopwords: Text with stopwords removed
    """
    stop_words = set(stopwords.words(lang))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def lemmatize_text(text):
    """
    Lemmatize text
    
    Args:
        text: Text to lemmatize
        
    Returns:
        lemmatized_text: Lemmatized text
    """
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

def preprocess_text(text, remove_stop=True, lemmatize=True):
    """
    Full preprocessing pipeline
    
    Args:
        text: Text to preprocess
        remove_stop: Whether to remove stopwords
        lemmatize: Whether to lemmatize
        
    Returns:
        preprocessed_text: Fully preprocessed text
    """
    # Clean text (lowercase, remove special chars, etc.)
    text = clean_text(text)
    
    # Remove stopwords if requested
    if remove_stop:
        text = remove_stopwords(text)
    
    # Lemmatize if requested
    if lemmatize:
        text = lemmatize_text(text)
    
    return text

def preprocess_dataset(texts, remove_stop=True, lemmatize=True):
    """
    Preprocess a list of texts
    
    Args:
        texts: List of texts to preprocess
        remove_stop: Whether to remove stopwords
        lemmatize: Whether to lemmatize
        
    Returns:
        preprocessed_texts: List of preprocessed texts
    """
    preprocessed_texts = []
    for text in texts:
        preprocessed_text = preprocess_text(text, remove_stop, lemmatize)
        preprocessed_texts.append(preprocessed_text)
    
    return preprocessed_texts