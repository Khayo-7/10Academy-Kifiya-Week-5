import re
import os
import sys
import pandas as pd
from typing import List, Tuple, Union, Dict, Optional

import sentencepiece as spm
from transformers import AutoTokenizer
from amseg.amharicSegmenter import AmharicSegmenter

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("tokenizer")

class Tokenizer:
    """
    Handles tokenization, alignment, and tokenizer initialization.
    Supports Hugging Face and custom tokenization methods.
    """

    def __init__(self, tokenizer: Union[AutoTokenizer, "AmharicSegmenter"], max_length: int = 128, label_id_mapping: Optional[Dict[str, int]] = None):
        """
        Initializes the Tokenizer with a tokenizer instance, max sequence length, and label mapping.

        Args:
            tokenizer (Union[AutoTokenizer, AmharicSegmenter]): Tokenizer instance.
            max_length (int): Maximum sequence length for tokenization.
            label_id_mapping (Optional[Dict[str, int]]): Mapping of string labels to integer IDs.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_id_mapping = label_id_mapping or {"O": 0}  # Default mapping for "O"
        logger.info(f"Tokenizer initialized with max length: {self.max_length}")

    def _validate_inputs(self, tokens: List[str], labels: Optional[List[str]] = None) -> None:
        """
        Validates input tokens and labels.

        Args:
            tokens (List[str]): List of tokens.
            labels (Optional[List[str]): NER labels corresponding to tokens.

        Raises:
            ValueError: If tokens or labels are invalid.
        """
        if not tokens:
            raise ValueError("Tokens must not be empty.")
        if labels is not None:
            if not labels:
                raise ValueError("Labels must not be empty.")
            if len(tokens) != len(labels):  # Check against the first list of labels
                raise ValueError("Tokens and labels must have the same length.")

    def _tokenize_input(self, tokens: List[str]) -> Dict:
        """
        Tokenizes input tokens using Hugging Face's tokenizer.

        Args:
            tokens (List[str]): List of tokens to tokenize.

        Returns:
            Dict: Tokenized output from Hugging Face tokenizer.
        """
        return self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors="pt" if self.tokenizer.is_fast else "np",  # NumPy fallback
        )

    def _convert_labels_to_ids(self, labels: List[str]) -> List[int]:
        """
        Converts string labels to integer IDs using the label_id_mapping.

        Args:
            labels (List[str]): List of string labels.

        Returns:
            List[int]: List of integer label IDs.
        """
        return [self.label_id_mapping.get(label, -100) for label in labels]

    def align_labels_with_tokens_hf(self, tokens: List[str], labels: List[str]) -> Tuple[List[str], List[str], List[int]]:
        """
        Aligns labels with tokenized tokens and returns aligned tokens, labels, and Hugging Face labels.

        Args:
            tokens (List[str]): List of input tokens.
            labels (List[str]): List of labels corresponding to the input tokens.

        Returns:
            Tuple[List[str], List[str], List[int]]: Aligned tokens, aligned labels, and Hugging Face labels.
        """
        # Tokenize the input tokens
        tokenized_inputs = self._tokenize_input(tokens)
        
        # Get word IDs for each token
        word_ids = tokenized_inputs.word_ids() if self.tokenizer.is_fast else self._get_word_ids_from_slow_tokenizer(tokenized_inputs)

        aligned_tokens, aligned_labels, hf_labels = [], [], []
        previous_word_idx = None
        input_ids = tokenized_inputs["input_ids"][0]  # Access the first sequence (batch size = 1)

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:  # Special tokens ([CLS], [SEP], [PAD])
                aligned_tokens.append(self.tokenizer.convert_ids_to_tokens(input_ids[idx].item()))  # Convert token ID to token
                # aligned_tokens.append(self.tokenizer.convert_ids_to_tokens(input_ids[idx].row()))  # Convert token ID to token
                aligned_labels.append("O")  # Use "O" for special tokens
                hf_labels.append(-100)  # Ignore special tokens during loss calculation
            
            else:
                if word_idx != previous_word_idx:  # First token of a word
                    aligned_tokens.append(tokens[word_idx])
                    aligned_labels.append(labels[word_idx])
                    hf_labels.append(self.label_id_mapping.get(labels[word_idx], -100))  # Map label to ID or use -100

                else:  # Subsequent tokens of the same word
                    aligned_tokens.append(self.tokenizer.convert_ids_to_tokens(input_ids[idx].item()))
                    # aligned_tokens.append(self.tokenizer.convert_ids_to_tokens(input_ids[idx].row()))
                    aligned_labels.append(labels[word_idx])  # Inherit the label from the first token
                    hf_labels.append(-100)  # Ignore during loss calculation

            previous_word_idx = word_idx

        # Add Hugging Face-compatible labels to tokenized_tokens
        tokenized_inputs["labels"] = hf_labels

        return aligned_tokens, aligned_labels, tokenized_inputs

    def _get_word_ids_from_slow_tokenizer(self, tokenized_inputs) -> List[Optional[int]]:
        """
        Mimics the behavior of `word_ids()` for slow tokenizers.
        This method aligns word indices with tokenized inputs for non-fast tokenizers.

        Args:
            tokenized_inputs: Tokenized output from the tokenizer.

        Returns:
            List[Optional[int]]: List of word IDs aligned with tokenized tokens.
        """
        word_ids = []
        current_word_idx = 0
        for token in tokenized_inputs["input_ids"][0]:
            token = self.tokenizer.convert_ids_to_tokens(token.item())  # Convert token ID to string
            if token.startswith("##"):  # Subword token
                word_ids.append(current_word_idx)
            elif token in self.tokenizer.all_special_tokens:  # Special token
                word_ids.append(None)
            else:  # New word
                word_ids.append(current_word_idx)
                current_word_idx += 1
        return word_ids


    def align_labels_with_tokens_generic(self, tokens: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """
        Aligns the original labels with the tokenized tokens.

        Args:
            tokens (List[str]): Original tokens.
            labels (List[str]): Labels corresponding to the original tokens.

        Returns:
            Tuple[List[str], List[str]]: Aligned tokens and aligned labels.

        Raises:
            Exception: If label alignment fails.
        """
        try:
            aligned_labels, aligned_tokens = [], []
            token_idx = 0

            for token, label in zip(tokens, labels):
                # Tokenize the current token to get its subtokens
                if hasattr(self.tokenizer, "amharic_tokenizer"):
                    subtokens = self.tokenizer.amharic_tokenizer(token)
                else:
                    subtokens = self.tokenizer.tokenize(token)

                # Add subtokens to aligned_tokens
                aligned_tokens.extend(subtokens)

                # Assign the label to the first subtoken and "O" to subsequent subtokens
                aligned_labels.append(label)
                aligned_labels.extend(["O"] * (len(subtokens) - 1))

                # Update token_idx to point to the next token
                token_idx += len(subtokens)

            logger.info("Tokens and labels aligned successfully.")
            return aligned_tokens, aligned_labels

        except Exception as e:
            logger.error(f"Error during label alignment: {e}")
            raise

    def tokenize_and_align_labels(self, tokens: List[str], labels: List[str], use_hf: bool = False) -> Union[Tuple[List[str], List[str]], Dict]:
        """
        Unified method to tokenize and align labels based on the selected approach.

        Args:
            tokens (List[str]): List of tokens.
            labels (List[str]): NER labels corresponding to tokens.
            use_hf (bool): Whether to use the Hugging Face approach.

        Returns:
            Union[Tuple[List[str], List[str]], Dict]: Tokenized tokens and aligned labels or Hugging Face-compatible inputs.

        Raises:
            Exception: If tokenization or alignment fails.
        """
        try:
            # Validate inputs
            self._validate_inputs(tokens, labels)

            if use_hf:
                return self.align_labels_with_tokens_hf(tokens, labels)
            else:
                aligned_tokens, aligned_labels = self.align_labels_with_tokens_generic(tokens, labels)
                return aligned_tokens, aligned_labels, aligned_tokens  # Return tokenized tokens for consistency
        except Exception as e:
            logger.error(f"Error during tokenization and label alignment: {e}")
            raise

# Utility functions
def tokenize_text(text: str, include_punctuations: bool = False) -> List[str]:
    """
    Tokenizes input text into tokens with optional punctuation inclusion.

    Args:
        text (str): The text to tokenize.
        include_punctuations (bool): Whether to include punctuation in tokens.

    Returns:
        List[str]: List of tokens from the text.

    Raises:
        Exception: If tokenization fails.
    """
    try:
        if include_punctuations:
            return re.findall(r'\S+', text)  # Include punctuation
        else:
            return re.findall(r'\w+', text)  # Exclude punctuation
    except Exception as e:
        logger.error(f"Error during text tokenization: {e}")
        raise


def initialize_auto_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Initializes an AutoTokenizer with a given model name.

    Args:
        model_name (str): The name of the model to use for the tokenizer.

    Returns:
        AutoTokenizer: An instance of AutoTokenizer.

    Raises:
        Exception: If initialization fails.
    """
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Error during AutoTokenizer initialization: {e}")
        raise


def initialize_amharic_segmenter(sent_punct: List[str] = [], word_punct: List[str] = []) -> "AmharicSegmenter":
    """
    Initializes an AmharicSegmenter.

    Args:
        sent_punct (List[str]): Sentence-level punctuation for segmentation.
        word_punct (List[str]): Word-level punctuation for segmentation.

    Returns:
        AmharicSegmenter: An instance of AmharicSegmenter.

    Raises:
        Exception: If initialization fails.
    """
    try:
        return AmharicSegmenter(sent_punct, word_punct)
    except Exception as e:
        logger.error(f"Error during AmharicSegmenter initialization: {e}")
        raise

def separate_tokens_and_labels(data: Union[pd.DataFrame, pd.Series], columns: List[str] = ['Tokens', 'Labels']) -> Tuple[List[str], List[str]]:
    """
    Separates tokens and labels from a DataFrame or Series of strings.

    Args:
        data (Union[pd.DataFrame, pd.Series]): A DataFrame or Series containing tokens and labels.
        columns (List[str], optional): A list of column names to use for the data. Defaults to ['Tokens', 'Labels'].

    Returns:
        Tuple[List[str], List[str]]: A tuple containing tokens and labels.

    Raises:
        ValueError: If 'columns' is empty, has more than 2 columns, or the input data type is unsupported.
        Exception: If an error occurs during the separation process.
    """
    try:
        logger.info("Starting to separate tokens and labels from the data.")

        # Validate columns
        if not columns or len(columns) != 2:
            raise ValueError("'columns' must contain exactly 2 column names.")

        all_tokens, all_labels = [], []

        if isinstance(data, pd.DataFrame):
            # Extract tokens and labels directly from the DataFrame
            if columns[0] not in data.columns or columns[1] not in data.columns:
                raise ValueError(f"Columns {columns} not found in the DataFrame.")
            
            all_tokens = data[columns[0]].explode().tolist()
            all_labels = data[columns[1]].explode().tolist()

        elif isinstance(data, pd.Series):
            # Convert Series of lists of tuples into a DataFrame
            temp_data = pd.DataFrame.from_records(data.explode().tolist(), columns=columns)
            all_tokens = temp_data[columns[0]].tolist()
            all_labels = temp_data[columns[1]].tolist()
        else:
            raise ValueError("Unsupported data type. Expected a DataFrame or Series.")

        logger.info("Finished separating tokens and labels from the data.")
        return all_tokens, all_labels

    except Exception as e:
        logger.error(f"Error during token and label separation: {e}")
        raise
    
def convert_tokens_labels_to_data(tokens, labels, return_dataframe=True):
    """
    Converts the given tokens and labels into a structured format, optionally returning a DataFrame.

    Args:
        tokens (list): A list of tokens extracted from text.
        labels (list): A list of labels corresponding to the tokens.
        return_dataframe (bool, optional): If True, returns a DataFrame. Defaults to True.

    Returns:
        Union[dict, pd.DataFrame]: A dictionary or DataFrame containing 'Tokens' and 'Labels' as keys or columns.
    """
    try:
        if return_dataframe:
            return pd.DataFrame({'Tokens': tokens, 'Labels': labels})
        
        return {'Tokens': tokens, 'Labels': labels}
    
    except Exception as e:
        logger.error(f"An error occurred while converting tokens and labels: {e}")
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
        data = convert_tokens_labels_to_data(tokens, labels)
        os.makedirs(out_dir, exist_ok=True)

        conll_output_path = os.path.join(out_dir, f"{filename}.conll")
        with open(conll_output_path, "w", encoding="utf-8") as f:
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
        logger.info(f"Labels saved to CoNLL file: {conll_output_path}")
    
        if save_csv:
            csv_output_path = os.path.join(out_dir, f"{filename}.csv")
            data.to_csv(csv_output_path, index=False)
            logger.info(f"Labels saved to CSV file: {csv_output_path}")
        if save_json:
            json_output_path = os.path.join(out_dir, f"{filename}.json")
            data.to_json(json_output_path, orient='records', lines=True, force_ascii=False)
            logger.info(f"Labels saved to JSON file: {json_output_path}")
        
    except Exception as e:
        logger.error(f"Error saving labels: {e}")
        raise

# def load_conll_data(file_path):
#     """
#     Loads CoNLL formatted data from a file and returns it as a DataFrame.

#     Args:
#         file_path (str): Path to the CoNLL formatted file.

#     Returns:
#         pd.DataFrame: DataFrame containing the loaded data with 'tokens' and 'labels' columns.
#     """
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             lines = f.readlines()
        
#         sentences = []
#         labels = []
#         current_sentence = []
#         current_labels = []
        
#         for line in lines:
#             line = line.strip()
#             if not line:  # Sentence boundary
#                 if current_sentence:
#                     sentences.append(current_sentence)
#                     labels.append(current_labels)
#                     current_sentence = []
#                     current_labels = []
#             else:
#                 token, tag = line.split()
#                 current_sentence.append(token)
#                 current_labels.append(tag)
        
#         return pd.DataFrame({"tokens": sentences, "labels": labels})
#     except FileNotFoundError:
#         logger.error(f"File not found: {file_path}")
#         raise
#     except Exception as e:
#         logger.error(f"Error loading CoNLL data: {e}")
#         raise