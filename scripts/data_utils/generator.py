from ast import Assign
import os
import re
import sys
import pandas as pd

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

def generate_labels(data, column, patterns):
    """
    Generates entity labels for tokens in the specified column of a DataFrame.

    Args:
        data (pd.DataFrame): The dataset.
        column (str): Column name containing the text messages.
        patterns (Dict[str, re.Pattern]): Patterns for labeling entities.

    Returns:
        pd.DataFrame: Updated DataFrame with token and label columns.
    """
    try:
        logger.info("Starting to generate labels.")
        labeled_data = []

        for message in data[column]:
            tokens = tokenize_text(message)
            labels = assign_entity_labels(tokens, patterns)
            labeled_data.append({"Message": message, "Tokens": tokens, "Labels": labels, "Tokens_Labels": list(zip(tokens, labels))})

        logger.info("Finished generating labels.")
        return pd.DataFrame(labeled_data)
    except Exception as e:
        logger.error(f"Error generating labels: {e}")
        raise

def save_conll_format(data, output_file, column=None, columns=None):
    """
    Saves labeled tokens and their entities in CoNLL format.

    Args:
        data (pd.DataFrame): DataFrame containing tokens and labels.
        output_file (str): Path to save the CoNLL formatted output.
        column (str, optional): The column containing the labeled data. Defaults to None.
        columns (tuple or list, optional): A tuple or list containing two column names for tokens and labels. Defaults to None.
    """
    try:
        if not column and not columns:
            raise ValueError("Either 'column' or 'columns' must be provided.")
        logger.info(f"Saving labeled data to {output_file} in CoNLL format.")
        with open(output_file, "w", encoding="utf-8") as f:
            for _, row in data.iterrows():
                if column:
                    for token, label in row[column]:
                        f.write(f"{token}\t{label}\n")
                elif columns:
                    token_column, label_column = columns
                    for token, label in zip(row[token_column], row[label_column]):
                        f.write(f"{token}\t{label}\n")
                f.write("\n")
        logger.info(f"Labeled data successfully saved to {output_file}.")
    except Exception as e:
        logger.error(f"Error saving CoNLL format: {e}")
        raise

def save_labels(tokens, labels, filename, out_dir, save_csv=True, save_json=False):
    """
    Saves the final tokens and labels to a CSV, JSON, and/or CoNLL file.

    Args:
        tokens (list): List of tokens.
        labels (list): List of labels.
        filename (str): Base name of the file to save.
        out_dir (str): Directory to save the file.
        save_csv (bool): If True, saves the data to a CSV file.
        save_json (bool): If True, saves the data to a JSON file.
    """
    try:
        df = pd.DataFrame({'Tokens': tokens, 'Labels': labels})
        os.makedirs(out_dir, exist_ok=True)

        conll_output_path = os.path.join(out_dir, f"{filename}.conll")
        with open(conll_output_path, "w", encoding="utf-8") as f:
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
        logger.info(f"Labels saved to CoNLL file: {conll_output_path}")
    
        if save_csv:
            csv_output_path = os.path.join(out_dir, f"{filename}.csv")
            df.to_csv(csv_output_path, index=False)
            logger.info(f"Labels saved to CSV file: {csv_output_path}")
        if save_json:
            json_output_path = os.path.join(out_dir, f"{filename}.json")
            df.to_json(json_output_path, orient='records', lines=True, force_ascii=False)
            logger.info(f"Labels saved to JSON file: {json_output_path}")
        
    except Exception as e:
        logger.error(f"Error saving labels: {e}")
        raise
    
def load_conll_data(file_path):
    """
    Loads CoNLL formatted data from a file and returns it as a DataFrame.

    Args:
        file_path (str): Path to the CoNLL formatted file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data with 'tokens' and 'labels' columns.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        sentences = []
        labels = []
        current_sentence = []
        current_labels = []
        
        for line in lines:
            line = line.strip()
            if not line:  # Sentence boundary
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                token, tag = line.split()
                current_sentence.append(token)
                current_labels.append(tag)
        
        return pd.DataFrame({"tokens": sentences, "labels": labels})
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading CoNLL data: {e}")
        raise