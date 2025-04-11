import logging
import os
import random
import nltk
from nltk.corpus import wordnet
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import pandas as pd
from tqdm import tqdm

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logging.warning(f"Error downloading NLTK resources: {e}")

def synonym_replacement(text, n=1):
    """Replace n words in the text with their synonyms"""
    try:
        words = nltk.word_tokenize(text)
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalnum()]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break

        return ' '.join(new_words)
    except Exception as e:
        logging.error(f"Error in synonym replacement: {e}")
        return text

def get_synonyms(word):
    """Get synonyms for a word"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and synonym.isalnum():
                synonyms.add(synonym)
    return synonyms

def back_translation(text, src='en', from_lang='fr'):
    """Translate text to another language and back to English"""
    try:
        aug = naw.BackTranslationAug(
            from_model_name='Helsinki-NLP/opus-mt-en-{}'.format(from_lang),
            to_model_name='Helsinki-NLP/opus-mt-{}-en'.format(from_lang)
        )
        augmented_text = aug.augment(text)
        return augmented_text
    except Exception as e:
        logging.error(f"Error in back translation: {e}")
        return text

def random_deletion(text, p=0.1):
    """Randomly delete words from the text with probability p"""
    try:
        words = nltk.word_tokenize(text)
        if len(words) == 1:
            return text
        
        # randomly delete words with probability p
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
                
        if len(new_words) == 0:  # if all words were deleted
            rand_int = random.randint(0, len(words)-1)
            new_words = [words[rand_int]]
            
        return ' '.join(new_words)
    except Exception as e:
        logging.error(f"Error in random deletion: {e}")
        return text

def random_swap(text, n=1):
    """Randomly swap n pairs of words in the text"""
    try:
        words = nltk.word_tokenize(text)
        if len(words) < 2:
            return text
        
        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
        return ' '.join(new_words)
    except Exception as e:
        logging.error(f"Error in random swap: {e}")
        return text

def apply_augmentations(texts, labels, techniques, multiplier=1):
    """
    Apply specified augmentation techniques to the dataset
    
    Args:
        texts: List of texts to augment
        labels: List of corresponding labels
        techniques: List of augmentation techniques to apply
        multiplier: Number of augmented examples to create per original example
        
    Returns:
        Tuple of augmented texts and labels
    """
    augmented_texts = []
    augmented_labels = []
    
    for i, (text, label) in enumerate(tqdm(zip(texts, labels), total=len(texts), desc="Augmenting data")):
        # Always keep the original example
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        # Apply each augmentation technique
        for _ in range(multiplier):
            for technique in techniques:
                try:
                    if technique == 'synonym':
                        aug_text = synonym_replacement(text)
                    elif technique == 'backtranslation':
                        aug_text = back_translation(text)
                    elif technique == 'deletion':
                        aug_text = random_deletion(text)
                    elif technique == 'swap':
                        aug_text = random_swap(text)
                    else:
                        logging.warning(f"Unknown augmentation technique: {technique}")
                        continue
                        
                    augmented_texts.append(aug_text)
                    augmented_labels.append(label)
                except Exception as e:
                    logging.error(f"Error applying {technique} to text {i}: {e}")
    
    logging.info(f"Created {len(augmented_texts)} examples from {len(texts)} original examples")
    return augmented_texts, augmented_labels

def augment_and_save(data_split, techniques, output_path, multiplier=1):
    """Augment data and save to CSV file"""
    texts, labels = data_split
    aug_texts, aug_labels = apply_augmentations(texts, labels, techniques, multiplier)
    
    # Create DataFrame and save
    df = pd.DataFrame({
        'text': aug_texts,
        'label': aug_labels
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Augmented data saved to {output_path}")
    
    return aug_texts, aug_labels