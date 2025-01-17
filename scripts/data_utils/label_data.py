# scripts/data_utils/label_data.py
import os
import sys
import pandas as pd

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import load_csv

logger = setup_logger("label_data")

DATA_PATH = os.path.join('..', '..', 'resources', 'data')
label_dir = os.path.join(DATA_PATH, 'labeled')
cleaned_data_path = os.path.join(DATA_PATH, 'cleaned', 'preprocessed.csv')

# Ensure the data directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# Define entity labels
ENTITY_LABELS = {
    '1': 'B-Product',
    '2': 'I-Product',
    '3': 'B-PRICE',
    '4': 'I-PRICE',
    '5': 'B-LOC',
    '6': 'I-LOC',
    '7': 'O'
}

def label_message(message):
    logger.info(f"Labeling message: {message}")
    tokens = message.split()
    labeled_tokens = []
    for token in tokens:
        logger.info(f"Labeling token: {token}")
        logger.info(f"Token: {token}")
        logger.info("Choose label:")
        for key, value in ENTITY_LABELS.items():
            logger.info(f"{key}: {value}")
        label = input("Enter label number: ")
        labeled_tokens.append(f"{token}\t{ENTITY_LABELS[label]}")
    return labeled_tokens

def save_to_conll(labeled_data, output_file):
    logger.info(f"Saving labeled data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in labeled_data:
            f.write('\n'.join(sentence) + '\n\n')

if __name__ == "__main__":
    labeled_data = []
    data = load_csv(cleaned_data_path)
    for message in data['cleaned_text'].head(30): 
        labeled_tokens = label_message(message)
        labeled_data.append(labeled_tokens)
    
    # Save labeled data in CoNLL format
    save_to_conll(labeled_data, os.path.join(label_dir, 'labeled_data.conll'))