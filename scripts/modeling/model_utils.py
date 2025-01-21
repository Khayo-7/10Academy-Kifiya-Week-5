import re
import os
import sys
import yaml
import torch
import random
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional

import sentencepiece as spm
from amseg.amharicSegmenter import AmharicSegmenter
from datasets import Dataset, DatasetDict, load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("model_utils")

# Utility functions
def tokenize_and_align_labels(examples: Dict, tokenizer: AutoTokenizer) -> Dict:
    """
    Tokenize inputs and align labels for token classification.

    Args:
        examples (Dict): Input examples containing tokens and labels.
        tokenizer (AutoTokenizer): Tokenizer to use for tokenization.

    Returns:
        Dict: Tokenized inputs with aligned labels.
    """
    try:
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True, max_length=128
        )
        labels = [
            [-100 if word_id is None else label[word_id] for word_id in tokenized_inputs.word_ids(batch_index=i)]
            for i, label in enumerate(examples["labels"])
        ]
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    except Exception as e:
        logger.error(f"Error during tokenization and label alignment: {e}")
        raise

def parse_conll(file_path: str) -> Dataset:
    """
    Parse a CoNLL file into a Hugging Face Dataset.

    Args:
        file_path (str): Path to the CoNLL file.

    Returns:
        Dataset: Hugging Face Dataset containing tokens and labels.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is improperly formatted.
    """
    try:
        sentences, labels = [], []
        temp_sent, temp_label = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if temp_sent:
                        sentences.append(temp_sent)
                        labels.append(temp_label)
                        temp_sent, temp_label = [], []
                else:
                    try:
                        word, tag = line.split()
                        temp_sent.append(word)
                        temp_label.append(tag)
                    except ValueError:
                        logger.error(f"Invalid line format in CoNLL file: {line}")
                        raise ValueError(f"Invalid line format in CoNLL file: {line}")
        if temp_sent:
            sentences.append(temp_sent)
            labels.append(temp_label)
        return Dataset.from_dict({"tokens": sentences, "labels": labels})
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error parsing CoNLL file: {e}")
        raise

def parse_conll_raw(file_path: str) -> List[List[str]]:
    """
    Parse a CoNLL file into a list of raw sentences, where each sentence is a list of lines.

    Args:
        file_path (str): Path to the CoNLL file.

    Returns:
        List[List[str]]: A list of sentences, where each sentence is a list of lines.
    """
    sentences = []
    current_sentence = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                current_sentence.append(line)
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
        if current_sentence:
            sentences.append(current_sentence)
    return sentences

def split_dataset(
    sentences: List[List[str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """
    Split a list of sentences into training, validation, and test sets.

    Args:
        sentences (List[List[str]]): List of sentences, where each sentence is a list of lines.
        train_ratio (float): Proportion of the dataset for training.
        val_ratio (float): Proportion for validation.
        test_ratio (float): Proportion for testing.
        random_seed (int): Seed for random shuffling.

    Returns:
        Tuple[List[List[str]], List[List[str]], List[List[str]]]: A tuple containing
            (train_data, val_data, test_data).

    Raises:
        ValueError: If the sum of `train_ratio`, `val_ratio`, and `test_ratio` is not equal to 1.
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    random.seed(random_seed)
    random.shuffle(sentences)

    total_sentences = len(sentences)
    train_size = int(total_sentences * train_ratio)
    val_size = int(total_sentences * val_ratio)

    train_data = sentences[:train_size]
    val_data = sentences[train_size : train_size + val_size]
    test_data = sentences[train_size + val_size:]

    return train_data, val_data, test_data

def save_conll_file(data: List[List[str]], file_path: str):
    """
    Save a list of sentences in CoNLL format to a file.

    Args:
        data (List[List[str]]): List of sentences, where each sentence is a list of lines.
        file_path (str): Path to save the data.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for sentence in data:
            f.writelines(line + "\n" for line in sentence)
            f.write("\n")  # Sentence boundary

def load_conll_data(data_dir: str, use_hf_load_dataset: bool = True) -> DatasetDict:
    """
    Load data from CoNLL files into a DatasetDict.

    Args:
        data_dir (str): Path to the directory containing CoNLL files.
        use_hf_load_dataset (bool): If True, use Hugging Face's `load_dataset` with the "conll2003" format.
                                     If False, use the custom `parse_conll` function.

    Returns:
        DatasetDict: Hugging Face DatasetDict containing train, val, and test sets.

    Raises:
        FileNotFoundError: If any of the CoNLL files are missing.
        Exception: If an error occurs during loading.
    """
    try:
        train_file = os.path.join(data_dir, "train.conll")
        val_file = os.path.join(data_dir, "val.conll")
        test_file = os.path.join(data_dir, "test.conll")

        if not all(os.path.exists(f) for f in [train_file, val_file, test_file]):
            raise FileNotFoundError(f"One or more CoNLL files are missing in {data_dir}")

        if use_hf_load_dataset:
            return load_dataset(
                "conll2003",
                data_files={
                    "train": train_file,
                    "validation": val_file,
                    "test": test_file,
                },
            )
        else:
            return DatasetDict(
                {
                    "train": parse_conll(train_file),
                    "val": parse_conll(val_file),
                    "test": parse_conll(test_file),
                }
            )

    except FileNotFoundError as e:
        logger.error(f"Missing CoNLL file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading CoNLL data: {e}")
        raise

def initialize_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Initialize an AutoTokenizer with a given model name.

    Args:
        model_name (str): The name of the model to use for the tokenizer.

    Returns:
        AutoTokenizer: An instance of AutoTokenizer.

    Raises:
        Exception: If initialization fails.
    """
    try:
        logger.info(f"Initializing AutoTokenizer with model name: {model_name}")
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Error during AutoTokenizer initialization: {e}")
        raise


def initialize_model(model_name: str, num_labels: int) -> AutoModelForTokenClassification:
    """
    Initialize a model for token classification.

    Args:
        model_name (str): The name of the model to use for token classification.
        num_labels (int): The number of labels for token classification.

    Returns:
        AutoModelForTokenClassification: An instance of AutoModelForTokenClassification.

    Raises:
        Exception: If initialization fails.
    """
    try:
        logger.info(f"Initializing model with model name: {model_name} and number of labels: {num_labels}")
        return AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        raise


def initialize_data_collator(tokenizer: AutoTokenizer) -> DataCollatorForTokenClassification:
    """
    Initialize a data collator for token classification.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for token classification.

    Returns:
        DataCollatorForTokenClassification: An instance of DataCollatorForTokenClassification.

    Raises:
        Exception: If initialization fails.
    """
    try:
        logger.info(f"Initializing data collator with tokenizer: {tokenizer}")
        return DataCollatorForTokenClassification(tokenizer)
    except Exception as e:
        logger.error(f"Error during data collator initialization: {e}")
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
        logger.info(f"Initializing AmharicSegmenter with sentence punctuation: {sent_punct} and word punctuation: {word_punct}")
        return AmharicSegmenter(sent_punct, word_punct)
    except Exception as e:
        logger.error(f"Error during AmharicSegmenter initialization: {e}")
        raise

def map_and_format_datasets(label_mapping: Dict[str, int]) -> ClassLabel:
    """
    Maps and formats datasets using the provided label mapping.

    Args:
        label_mapping (Dict[str, int]): A dictionary mapping string labels to integer IDs.

    Returns:
        ClassLabel: A ClassLabel instance with the provided label mapping.

    Raises:
        ValueError: If the label mapping is invalid or empty.
    """
    if not label_mapping:
        raise ValueError("Label mapping cannot be empty.")
    try:
        return ClassLabel(names=label_mapping)
    except Exception as e:
        logger.error(f"Error during dataset mapping and formatting: {e}")
        raise

def convert_df_to_dataset(data: pd.DataFrame) -> Dataset:
    """
    Converts a pandas DataFrame to a Hugging Face Dataset.

    Args:
        data (pd.DataFrame): The DataFrame to convert.

    Returns:
        Dataset: The converted dataset.
    """
    try:
        return Dataset.from_pandas(data)
    except Exception as e:
        logger.error(f"Error converting DataFrame to dataset: {e}")
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

        if not columns or len(columns) != 2:
            raise ValueError("'columns' must contain exactly 2 column names.")

        all_tokens, all_labels = [], []

        if isinstance(data, pd.DataFrame):
            if columns[0] not in data.columns or columns[1] not in data.columns:
                raise ValueError(f"Columns {columns} not found in the DataFrame.")
            
            all_tokens = data[columns[0]].explode().tolist()
            all_labels = data[columns[1]].explode().tolist()

        elif isinstance(data, pd.Series):
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

def save_labels(tokens: List[str], labels: List[str], filename: str, out_dir: str, save_csv: bool = True, save_json: bool = False):
    """
    Saves the final tokens and labels to a CSV, JSON, and/or CoNLL file.

    Args:
        tokens (List[str]): List of tokens.
        labels (List[str]): List of labels.
        filename (str): Base name of the file to save.
        out_dir (str): Directory to save the file.
        save_csv (bool): If True, saves the data to a CSV file.
        save_json (bool): If True, saves the data to a JSON file.
    """
    def convert_tokens_labels_to_data(tokens: List[str], labels: List[str], return_dataframe: bool = True) -> Union[pd.DataFrame, Dict]:
        """
        Converts tokens and labels into a DataFrame or dictionary.

        Args:
            tokens (List[str]): List of tokens.
            labels (List[str]): List of labels.
            return_dataframe (bool): If True, returns a DataFrame; otherwise, returns a dictionary.

        Returns:
            Union[pd.DataFrame, Dict]: DataFrame or dictionary containing tokens and labels.
        """
        try:
            if return_dataframe:
                return pd.DataFrame({'Tokens': tokens, 'Labels': labels})
            return {'Tokens': tokens, 'Labels': labels}
        except Exception as e:
            logger.error(f"An error occurred while converting tokens and labels: {e}")
            raise

    try:
        os.makedirs(out_dir, exist_ok=True)

        conll_output_path = os.path.join(out_dir, f"{filename}.conll")
        with open(conll_output_path, "w", encoding="utf-8") as f:
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
        logger.info(f"Labels saved to CoNLL file: {conll_output_path}")
    
        data = convert_tokens_labels_to_data(tokens, labels)
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

