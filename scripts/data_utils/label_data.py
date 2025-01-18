# scripts/data_utils/label_data.py
import os
import sys
import pandas as pd

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import load_csv

logger = setup_logger("label_data")

DATA_PATH = os.path.join('..', 'resources', 'data')
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
    """
    Labels a given message by asking the user to input labels for each token.
    
    Args:
        message (str): The message to be labeled.
    
    Returns:
        list: A list of labeled tokens.
    """
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
    """
    Saves labeled data to a file in CoNLL format.
    
    Args:
        labeled_data (list): A list of labeled data.
        output_file (str): The path to the output file.
    """
    logger.info(f"Saving labeled data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in labeled_data:
            f.write('\n'.join(sentence) + '\n\n')

def label_data(data, column='cleaned_text'):
    """
    Labels data in a given column of a DataFrame.
    
    Args:
        data (pd.DataFrame): The DataFrame containing the data to be labeled.
        column (str, optional): The column name in the DataFrame to label. Defaults to 'cleaned_text'.
    
    Returns:
        list: A list of labeled data.
    """
    labeled_data = []
    for message in data[column]: 
        labeled_tokens = label_message(message)
        labeled_data.append(labeled_tokens)
    
    return labeled_data

if __name__ == "__main__":
    data = load_csv(cleaned_data_path)
    labeled_data = label_data(data, 'cleaned_text')
    
    # Save labeled data in CoNLL format
    save_to_conll(labeled_data, os.path.join(label_dir, 'labeled_data.conll'))