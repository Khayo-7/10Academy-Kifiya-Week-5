import os
import sys
import pandas as pd
from transformers import AutoTokenizer
from amseg.amharicSegmenter import AmharicSegmenter
from datasets import load_dataset
from typing import List, Union
import sentencepiece as spm

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
logger = setup_logger("tokenizer")

def read_and_process_lines(file_path: str) -> pd.DataFrame:
    """
    Reads a file and processes its lines into a DataFrame.
    
    Args:
    - file_path (str): The path to the file to be read.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the processed lines.
    """
    logger.info(f"Starting to read and process lines from file: {file_path}")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [line.strip().split('\t') for line in lines]  
    logger.info(f"Finished reading and processing lines from file: {file_path}")
    return pd.DataFrame(data)

def separate_tokens_and_labels(data: pd.Series) -> tuple:
    """
    Separates tokens and labels from a Series of strings.
    
    Args:
    - data (pd.Series): A Series of strings containing tokens and labels.
    
    Returns:
    - tuple: A tuple containing two lists: tokens and labels.
    """
    logger.info("Starting to separate tokens and labels from the data.")
    processed_data = data.str.split().tolist() 
    tokens = [item[0] for item in processed_data]  # Extract tokens
    labels = [item[1] for item in processed_data]  # Extract labels
    logger.info("Finished separating tokens and labels from the data.")
    return tokens, labels

def initialize_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Initializes a tokenizer with a given model name.
    
    Args:
    - model_name (str): The name of the model to use for the tokenizer.
    
    Returns:
    - AutoTokenizer: An instance of AutoTokenizer.
    """
    logger.info(f"Starting to initialize tokenizer with model name: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Finished initializing tokenizer with model name: {model_name}")
    return tokenizer

def tokenize_and_align_labels(segmenter: Union[AutoTokenizer, AmharicSegmenter], tokens: list, labels: list) -> tuple:
    """
    Aligns tokens with their corresponding labels.
    
    Args:
    - segmenter (Union[AutoTokenizer, AmharicSegmenter]): The segmenter to use for tokenization.
    - tokens (list): A list of tokens to be aligned.
    - labels (list): A list of labels corresponding to the tokens.
    
    Returns:
    - tuple: A tuple containing two lists: aligned tokens and aligned labels.
    """
    logger.info("Starting to align tokens with their corresponding labels.")
    aligned_tokens = []
    aligned_labels = []

    for word, label in zip(tokens, labels):
        if isinstance(segmenter, AutoTokenizer):
            tokenized_word = segmenter.tokenize(word)
        else:
            tokenized_word = segmenter.amharic_tokenizer(word)
        aligned_tokens.extend(tokenized_word)

        # Assign the label to the first subtoken and 'O' to subsequent subtokens
        aligned_labels.extend([label] + ['O'] * (len(tokenized_word) - 1))

    logger.info("Finished aligning tokens with their corresponding labels.")
    return aligned_tokens, aligned_labels

def save_to_csv(tokens: list, labels: list, file_path: str) -> None:
    """
    Saves aligned tokens and labels to a CSV file.
    
    Args:
    - tokens (list): A list of aligned tokens.
    - labels (list): A list of aligned labels.
    - file_path (str): The path to the file where the data will be saved.
    """
    logger.info(f"Starting to save aligned tokens and labels to CSV file: {file_path}")
    output_df = pd.DataFrame({'Token': tokens, 'Label': labels})
    output_df.to_csv(file_path, index=False)
