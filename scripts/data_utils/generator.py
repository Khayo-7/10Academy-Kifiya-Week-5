import os
import re
import sys
import pandas as pd
from typing import Dict
from multiprocessing import Pool, cpu_count

# Setup logger
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.modeling.tokenizer import tokenize_text

logger = setup_logger("generator")

# Define regex patterns for entity labeling
PATTERNS = {
    "Product": re.compile(r"\b(ወርቅ|ሸማት|ኪሳራ|ሻርቤት}|በቃል|ፍሬ|ነገር|አካሉ|ቤት)\b"),
    "PRICE": re.compile(r"\b(በ|ዋጋ)\s?\d+\s?ብር\b"),
    "LOC": re.compile(r"\b(አዲስ\s?አበባ|ቦሌ|መቐለ|አዳማ|ሲዳማ|ሱቅ|ነዋሪ)\b"),
}

def assign_entity_labels(tokens, patterns=PATTERNS):
    """
    Assigns CoNLL-compliant labels to tokens based on the specified entity patterns.

    Args:
        tokens (List[str]): List of tokens from a message.
        patterns (Dict[str, re.Pattern]): Entity patterns to label tokens.

    Returns:
        List[str]: List of labels corresponding to the tokens in CoNLL format.
    """
    try:
        logger.info("Starting to assign entity labels.")
        # Default to "O" (Outside any entities)
        labels = ["O"] * len(tokens)

        # Reconstruct text for pattern matching
        text = " ".join(tokens)

        # Track positions of matches to prevent overlapping labeling
        used_positions = set()

        # Apply patterns to label entities
        for entity_type, pattern in patterns.items():

            # Iterate over all matches for this entity type
            for match in pattern.finditer(text):

                # Determine token start/end positions of the match
                match_start = len(text[:match.start()].split())  # Start token index
                match_end = len(text[:match.end()].split()) - 1  # End token index

                # Skip if any part of the match overlaps an already-labeled position
                if any(pos in used_positions for pos in range(match_start, match_end + 1)):
                    continue

                # Assign B-ENTITY for first token, I-ENTITY for subsequent tokens
                labels[match_start] = f"B-{entity_type}"
                for i in range(match_start + 1, match_end + 1):
                    labels[i] = f"I-{entity_type}"

                # Mark these positions as used
                used_positions.update(range(match_start, match_end + 1))

        logger.info("Finished assigning entity labels.")
        return labels
    except Exception as e:
        logger.error(f"Error assigning entity labels: {e}")
        raise

def tokenize_and_label(message: str, patterns: Dict[str, re.Pattern]) -> tuple:
    """
    Tokenizes a message and assigns entity labels to the tokens.

    Args:
        message (str): The input message.
        patterns (Dict[str, re.Pattern]): Patterns for labeling entities.

    Returns:
        tuple: A tuple containing (tokens, labels, tokens_labels).
    """
    tokens = tokenize_text(message)
    labels = assign_entity_labels(tokens, patterns)
    tokens_labels = list(zip(tokens, labels))
    return tokens, labels, tokens_labels

def generate_labels(data: pd.DataFrame, column: str, patterns: Dict[str, re.Pattern]) -> pd.DataFrame:
    """
    Generates entity labels for tokens in the specified column of a DataFrame and appends the results to the DataFrame.

    Args:
        data (pd.DataFrame): The input dataset.
        column (str): Column name containing the text messages.
        patterns (Dict[str, re.Pattern]): Patterns for labeling entities.

    Returns:
        pd.DataFrame: Updated DataFrame with appended token and label columns.
    """
    try:
        logger.info("Starting to generate labels.")

        data = data.copy()
        # Use multiprocessing to process messages in parallel
        with Pool(cpu_count()) as pool:
            results = pool.starmap(
                tokenize_and_label,
                [(message, patterns) for message in data[column]]
            )

        # Unpack results into separate lists
        tokens_list, labels_list, tokens_labels_list = zip(*results)

        # Append the new columns to the input DataFrame
        data["Tokens"] = tokens_list
        data["Labels"] = labels_list
        data["Tokens_Labels"] = tokens_labels_list

        logger.info("Finished generating labels and appending to DataFrame.")
        return data

    except Exception as e:
        logger.error(f"Error generating labels: {e}")
        raise
