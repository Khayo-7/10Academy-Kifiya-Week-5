import re
import os
import sys
import yaml
import torch
import random
import pandas as pd
from typing import List, Dict, Tuple, Union, Generator

import sentencepiece as spm
from datasets import ClassLabel
from amseg.amharicSegmenter import AmharicSegmenter
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.modeling.tokenizer import Tokenizer
from scripts.data_utils.loaders import load_json, save_json

logger = setup_logger("model_utils")

# Utility functions
def parse_conll(file_path: str) -> Generator[Dict[str, List[str]], None, None]:
    """
    Parse a CoNLL file into a generator of dictionaries containing tokens and labels.

    Args:
        file_path (str): Path to the CoNLL file.

    Yields:
        Generator[Dict[str, List]]: A generator of dictionaries containing tokens and labels.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is improperly formatted.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tokens, labels = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        yield {
                            "tokens": tokens, 
                            "labels": labels
                        }
                        tokens, labels = [], []
                else:
                    try:
                        word, tag = line.split()
                        tokens.append(word)
                        labels.append(tag)
                    except ValueError:
                        logger.error(f"Invalid line format in CoNLL file: {line}")
                        raise ValueError(f"Invalid line format in CoNLL file: {line}")
            if tokens:
                yield {
                    "tokens": tokens, 
                    "labels": labels
                }
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error parsing CoNLL file: {e}")
        raise

def parse_conll_raw(file_path: str) -> Generator[List[List[str]], None, None]:
    """
    Parse a CoNLL file into a list of raw sentences, where each sentence is a list of lines.

    Args:
        file_path (str): Path to the CoNLL file.

    Returns:
        List[List[str]]: A list of sentences, where each sentence is a list of lines.
    """
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
            else:
                if sentences:
                    yield sentences

                    sentences = []
        if sentences:
            yield sentences

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


def split_and_save_datasets(filepath, data_dir: str):

    os.makedirs(data_dir, exist_ok=True)
    train_filepath = os.path.join(data_dir, "train.conll")
    val_filepath = os.path.join(data_dir, "val.conll")
    test_filepath = os.path.join(data_dir, "test.conll")
    
    sentences = [sentence for sentence in parse_conll_raw(filepath)]

    train_data, val_data, test_data = split_dataset(sentences)
    save_conll_file(train_data, train_filepath)
    save_conll_file(val_data, val_filepath)
    save_conll_file(test_data, test_filepath)

def load_conll_datasets(data_dir: str, use_hf: bool = True, format="conll") -> DatasetDict:
    """
    Load data from CoNLL files into a DatasetDict.

    Args:
        data_dir (str): Path to the directory containing CoNLL files.
        use_hf (bool): If True, use Hugging Face's `load_dataset` with a custom script.
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
        script_file = "conll_script.py"

        if not all(os.path.exists(f) for f in [train_file, val_file, test_file]):
            raise FileNotFoundError(f"One or more CoNLL files are missing in {data_dir}")

        if use_hf:
            LABEL_LIST = ["O", "B-Product", "I-Product", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE"]
            # Load custom CoNLL files using Hugging Face's `load_dataset`
            if format == "conll":
                return load_dataset(script_file, data_files = {
                        "train": train_file,
                        "validation": train_file,
                        "test": train_file
                    },
                    trust_remote_code=True,  # Allow execution of custom code
                )
            # elif format == "csv":
            #     # Load custom CoNLL files using Hugging Face's `load_dataset`
            #     return load_dataset("csv", data_files={
            #             "train": train_file,
            #             "validation": val_file,
            #             "test": test_file,
            #         },
            #         trust_remote_code=True  # Allow execution of custom code
            #     )
            # elif format == "json":
            #     return load_dataset("json", data_files={
            #             "train": train_file,
            #             "validation": val_file,
            #             "test": test_file,
            #         },
            #         trust_remote_code=True  # Allow execution of custom code
            #     )
        else:

            # Load custom CoNLL dataset
            # return DatasetDict.load_from_disk(file_path)
            
            # Load datasets
            train_data = [data for data in parse_conll(train_file)]
            val_data = [data for data in parse_conll(val_file)]
            test_data = [data for data in parse_conll(test_file)]

            return DatasetDict(
                {
                    "train": convert_data_to_dataset(train_data),
                    "val": convert_data_to_dataset(val_data),
                    "test": convert_data_to_dataset(test_data),
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

def convert_data_to_dataset(data: Union[pd.DataFrame, Dict, List]) -> Dataset:
    """
    Converts data to a Hugging Face Dataset based on its type.

    Args:
        data (Union[pd.DataFrame, Dict, List]): The data to convert.

    Returns:
        Dataset: The converted dataset.

    Raises:
        Exception: If the conversion fails.
    """
    try:
        if isinstance(data, pd.DataFrame):
            return Dataset.from_pandas(data)
        elif isinstance(data, dict):
            return Dataset.from_dict(data)
        elif isinstance(data, list):
            return Dataset.from_list(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}. Expected DataFrame, Dict, or List.")
    except Exception as e:
        logger.error(f"Error converting data to dataset: {e}")
        raise

def separate_tokens_and_labels(
    data: Union[pd.DataFrame, pd.Series], 
    columns: List[str] = ['Tokens', 'Labels']
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Separates tokens and labels from a DataFrame or Series of strings.

    Args:
        data (Union[pd.DataFrame, pd.Series]): A DataFrame or Series containing tokens and labels.
        columns (List[str], optional): A list of column names to use for the data. Defaults to ['Tokens', 'Labels'].

    Returns:
        Tuple[List[List[str]], List[List[str]]]: A tuple containing tokens and labels as lists of lists (one list per sentence).

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
            
            # Ensure tokens and labels are lists of lists (one list per sentence)
            all_tokens = data[columns[0]].tolist()
            all_labels = data[columns[1]].tolist()

        elif isinstance(data, pd.Series):
            # Convert Series to DataFrame
            temp_data = pd.DataFrame.from_records(data.explode().tolist(), columns=columns)
            all_tokens = temp_data[columns[0]].tolist()
            all_labels = temp_data[columns[1]].tolist()
        else:
            raise ValueError("Unsupported data type. Expected a DataFrame or Series.")

        # Ensure tokens and labels are lists of lists (one list per sentence)
        if not all(isinstance(tokens, list) for tokens in all_tokens):
            all_tokens = [tokens if isinstance(tokens, list) else [tokens] for tokens in all_tokens]
        if not all(isinstance(labels, list) for labels in all_labels):
            all_labels = [labels if isinstance(labels, list) else [labels] for labels in all_labels]

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


def load_and_tokenize_dataset(data_path: str, tokenizer, ner_tags: List, max_length: int, use_hf: bool = False) -> DatasetDict:
    """
    Load and tokenize the dataset.

    Args:
        data_path (str): Path to the dataset.
        tokenizer: Tokenizer to use.
        ner_tags (List): List of NER tags.
        max_length (int): Maximum length for tokenization.
        use_hf (bool): Whether to use Hugging Face for loading dataset.

    Returns:
        DatasetDict: Tokenized dataset.
    """
    logger.info("[INFO] Loading dataset...")
    dataset = load_conll_datasets(data_path, use_hf=use_hf)
    logger.info("[INFO] Finished loading dataset.")

    # Tokenize datasets
    tokenify = Tokenizer(tokenizer, ner_tags=ner_tags, max_length=max_length)

    logger.info("[INFO] Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenify.tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,  # To save space
        # num_proc=4,  # Multiple processes for efficiency
    )
    logger.info("[INFO] Finished tokenizing dataset.")

    return tokenized_datasets

def save_tokenized_dataset(tokenized_datasets: DatasetDict, tokenized_dir: str) -> None:
    """
    Save tokenized dataset to disk.

    Args:
        tokenized_datasets (DatasetDict): Tokenized dataset to save.
        tokenized_dir (str): Directory to save tokenized dataset.
    """
    logger.info("[INFO] Saving tokenized dataset...")
    tokenized_datasets.save_to_disk(tokenized_dir)
    logger.info("[INFO] Finished saving tokenized dataset.")

def load_tokenized_datasets(tokenized_dir: str) -> DatasetDict:
    """
    Load tokenized dataset from disk if it exists.

    Args:
        tokenized_dir (str): Directory to load tokenized dataset.

    Returns:
        DatasetDict: Tokenized dataset.
    """
    logger.info("[INFO] Loading tokenized dataset from disk.")
    tokenized_datasets = DatasetDict.load_from_disk(tokenized_dir)
    return tokenized_datasets

def load_tokenized_dataset(tokenized_dir: str) -> DatasetDict:
    """
    Load tokenized dataset from disk if it exists.

    Args:
        tokenized_dir (str): Directory to load tokenized dataset.

    Returns:
        DatasetDict: Tokenized dataset.
    """
    logger.info("[INFO] Loading tokenized dataset from disk.")
    tokenized_datasets = Dataset.load_from_disk(tokenized_dir)
    # tokenized_datasets = load_from_disk(tokenized_dir)
    return tokenized_datasets

def save_metadata(metadata: Dict, metadata_filepath: str) -> None:
    """
    Save metadata to disk.

    Args:
        metadata (Dict): Metadata to save.
        metadata_filepath (str): Path to save metadata.
    """
    
    logger.info("[INFO] Saving metadata...")
    save_json(metadata, metadata_filepath, use_pandas=False)
    logger.info("[INFO] Finished saving metadata.")

def load_metadata(metadata_filepath: str) -> Dict:
    """
    Load metadata from disk if it exists.

    Args:
        metadata_filepath (str): Path to load metadata.

    Returns:
        Dict: Metadata.
    """
    logger.info("[INFO] Loading metadata from disk.")
    metadata = load_json(metadata_filepath)
    return metadata


def tokenize_and_align_labels(sentences: Dict, tokenizer: AutoTokenizer, label_id_mapping: Dict[str, int]) -> Dict:
    """
    Tokenize inputs and align labels for token classification.
    """
    
    try:
        tokenized_inputs = tokenizer(
            sentences["tokens"], truncation=True, is_split_into_words=True, max_length=128
        )
        labels = [
            [-100 if word_id is None else label[word_id] for word_id in tokenized_inputs.word_ids(batch_index=i)]
        #     [-100 if word_id is None else label_id_mapping.get(label[word_id], -100) for word_id in tokenized_inputs.word_ids(batch_index=i)]
            for i, label in enumerate(sentences["labels"])
        ]

        # labels = []
        # for i, label in enumerate(sentences["labels"]):
        #     word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word
        #     previous_word_idx = None
        #     label_ids = []
        #     for word_idx in word_ids:
        #         if word_idx is None:
        #             label_ids.append(-100)  # Special token
        #         elif word_idx != previous_word_idx:
        #             label_ids.append(label[word_idx])  # New word
        #         else:
        #             label_ids.append(-100)  # Same word, subsequent token
        #         previous_word_idx = word_idx
        #     labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    except Exception as e:
        logger.error(f"Error during tokenization and label alignment: {e}")
        raise


def create_label_id_mapping(labels: List[str]) -> ClassLabel:
    """
    Creates a ClassLabel instance from the provided label mapping.

    Args:
        labels (List[str]): List of string labels.

    Returns:
        ClassLabel: A ClassLabel instance for label-to-ID mapping.

    Raises:
        ValueError: If the label list is empty.
    """
    if not labels:
        raise ValueError("Label list cannot be empty.")
    try:
        class_label = ClassLabel(names=labels)
        return {label: class_label.str2int(label) for label in labels}
    except Exception as e:
        logger.error(f"Error when creating label mapping: {e}")
        raise
