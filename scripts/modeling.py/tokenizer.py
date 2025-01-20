import re
import os
import sys
import pandas as pd
from typing import Union
import sentencepiece as spm
from sympy import Segment
from transformers import AutoTokenizer
from amseg.amharicSegmenter import AmharicSegmenter

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
logger = setup_logger("tokenizer")

def tokenize_text(text: str):
    """
    Tokenizes input text into tokens while respecting punctuation and spaces.

    Args:
        text (str): The text to tokenize.

    Returns:
        List[str]: List of tokens from the text.
    """
    return re.findall(r'\S+', text)
    return re.findall(r'\w+', text)
    # return text.split()

def initialize_auto_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Initializes an AutoTokenizer with a given model name.
    
    Args:
    - model_name (str): The name of the model to use for the tokenizer.
    
    Returns:
    - AutoTokenizer: An instance of AutoTokenizer.
    """
    logger.info(f"Starting to initialize AutoTokenizer with model name: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Finished initializing AutoTokenizer with model name: {model_name}")
    return tokenizer

def initialize_amharic_segmenter(sent_punct=[], word_punct=[]) -> AmharicSegmenter:
    """
    Initializes an AmharicSegmenter.
    
    Returns:
    - AmharicSegmenter: An instance of AmharicSegmenter.
    """
    logger.info("Starting to initialize AmharicSegmenter")
    segmenter = AmharicSegmenter(sent_punct, word_punct)
    logger.info("Finished initializing AmharicSegmenter")
    return segmenter

def separate_tokens_and_labels(data: pd.Series) -> tuple:
    """
    Separates tokens and labels from a Series of strings.
    
    Args:
    - data (pd.Series): A Series of strings containing tokens and labels.
    
    Returns:
    - tuple: A tuple containing two lists: tokens and labels.
    """
    logger.info("Starting to separate tokens and labels from the data.")
    # data = [[(የእናት, O), (ጡት, O), (ጫፍ, O)], 
    #          [(ማራዘሚያ, O), (ዋጋ,I-Product]
                # ...]
    all_tokens, all_labels = [], []
    for item in data:
        if item:  # Check if item is not empty
            tokens, labels = zip(*item)  # Separate tokens and labels using zip for each item
            all_tokens.extend(tokens)
            all_labels.extend(labels)
        else:
            logger.warning("Empty item found in data. Skipping...")
    logger.info("Finished separating tokens and labels from the data.")
    return list(all_tokens), list(all_labels)

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

    for token, label in zip(tokens, labels):
        if isinstance(segmenter, AmharicSegmenter):
            tokenized_token = segmenter.amharic_tokenizer(token)
        else:
            tokenized_token = segmenter.tokenize(token)

        aligned_tokens.extend(tokenized_token)

        # Assign the label to the first subtoken and 'O' to subsequent subtokens
        aligned_labels.extend([label] + ['O'] * (len(tokenized_token) - 1))

    logger.info("Finished aligning tokens with their corresponding labels.")
    return aligned_tokens, aligned_labels
