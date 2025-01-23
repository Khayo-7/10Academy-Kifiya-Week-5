import os
import csv
import yaml
import json
import joblib 
import pandas as pd
from typing import List, Dict, Union, Optional, Any
from scripts.utils.logger import setup_logger

# Setup logger for data_loader
logger = setup_logger("data_loader")

def load_yml(file_path: str):
    """
    Load a YAML file into a Python object.

    Args:
        file_path (str): Path to the YAML file.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If an error occurs during loading.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        logger.info(f"Loading data from YAML file: {file_path}")
        with open(file_path, mode='r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        logger.info(f"Loaded YAML data from {file_path}.")
        return data
    except Exception as e:
        logger.error(f"Error loading YAML from {file_path}: {e}")
        raise

def load_csv(file_path: str, delimiter: str = ',', use_pandas: bool = True) -> Union[List[Dict[str, str]], pd.DataFrame]:
    """
    Load a CSV file into a list of dictionaries or a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        delimiter (str, optional): Delimiter to use. Defaults to ','.
        use_pandas (bool, optional): Whether to use pandas. Defaults to True.

    Returns:
        Union[List[Dict[str, str]], pd.DataFrame]: Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If an error occurs during loading.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        logger.info(f"Loading data from CSV file: {file_path}")
        if use_pandas:
            data = pd.read_csv(file_path, delimiter=delimiter)
            logger.info(f"Loaded {len(data)} rows from {file_path} using pandas.")
        else:
            data = []
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter=delimiter)
                for row in reader:
                    data.append(row)
            logger.info(f"Loaded {len(data)} rows from {file_path} without pandas.")
        return data
    except Exception as e:
        logger.error(f"Error loading CSV from {file_path}: {e}")
        raise

def save_csv(data: Union[List[Dict[str, str]], pd.DataFrame], output_path: str, use_pandas: bool = True) -> None:
    """
    Saves data to a CSV file.

    Args:
        data (Union[List[Dict[str, str]], pd.DataFrame]): Data to save.
        output_path (str): Path to save the CSV file.
        use_pandas (bool, optional): Whether to use pandas. Defaults to True.

    Raises:
        Exception: If an error occurs during saving.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if use_pandas:
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_path, index=False)
            else:
                pd.DataFrame(data).to_csv(output_path, index=False)
            logger.info(f"Data saved to {output_path} using pandas.")
        else:
            with open(output_path, mode='w', encoding='utf-8', newline='') as file:
                if data:
                    writer = csv.DictWriter(file, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
            logger.info(f"Data saved to {output_path} without pandas.")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")
        raise

def load_json(file_path: str, use_pandas: bool = False) -> Union[List[Dict], Dict, pd.DataFrame]:
    """
    Load a JSON file into a Python object or a pandas DataFrame.

    Args:
        file_path (str): Path to the JSON file.
        use_pandas (bool, optional): Whether to use pandas. Defaults to False.

    Returns:
        Union[List[Dict], Dict, pd.DataFrame]: Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If an error occurs during loading.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        logger.info(f"Loading data from JSON file: {file_path}")
        with open(file_path, mode='r', encoding='utf-8') as file:
            data = json.load(file)
        if use_pandas:
            data = pd.DataFrame(data)
            logger.info(f"Loaded JSON data from {file_path} using pandas.")
        else:
            logger.info(f"Loaded JSON data from {file_path} without pandas.")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        raise

def save_json(data, output_path: str, use_pandas: bool = True) -> None:
    """
    Saves data to a JSON file.

    Args:
        data: Data to save.
        output_path (str): Path to save the JSON file.
        use_pandas (bool, optional): Whether to use pandas. Defaults to True.

    Raises:
        Exception: If an error occurs during saving.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if use_pandas:
            if isinstance(data, pd.DataFrame):
                data.to_json(output_path, orient="records", lines=True, force_ascii=False)
            else:
                pd.DataFrame(data).to_json(output_path, orient="records", lines=True, force_ascii=False)
            logger.info(f"Data saved to {output_path} using pandas.")
        else:
            with open(output_path, mode='w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            logger.info(f"Data saved to {output_path} without pandas.")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")
        raise

def load_pickle(file_path: str) -> Any:
    """
    Load a pickle file into a Python object.

    Args:
        file_path (str): Path to the pickle file.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If an error occurs during loading.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        logger.info(f"Loading data from pickle file: {file_path}")
        data = joblib.load(file_path)
        logger.info(f"Loaded pickle data from {file_path}.")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle from {file_path}: {e}")
        raise

def save_pickle(data: Any, output_path: str) -> None:
    """
    Saves data to a pickle file.

    Args:
        data (Any): Data to save.
        output_path (str): Path to save the pickle file.

    Raises:
        Exception: If an error occurs during saving.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(data, output_path)
        logger.info(f"Data saved to {output_path} using pickle.")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")
        raise

def load_conll(file_path: str, columns=["tokens", "labels"], use_pandas: bool = True) -> Union[List[Dict[str, List[str]]], pd.DataFrame]:
    """
    Loads data from a CoNLL file into a list of dictionaries or a pandas DataFrame.

    Args:
        file_path (str): The path to the CoNLL file to be loaded.
        columns (list, optional): A list of column names to use for the data. Defaults to ["Tokens", "Labels"].
        use_pandas (bool, optional): A flag indicating whether to use pandas for data loading. Defaults to True.

    Returns:
        Union[List[Dict[str, List[str]]], pd.DataFrame]: The loaded data, either as a list of dictionaries or a pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: If an error occurs during the loading process.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        logger.info(f"Loading data from CoNLL file: {file_path}")
        data, tokens, labels = [], [], []
        tokens_column, labels_column = columns

        with open(file_path, mode='r', encoding='utf-8') as file:
            
            # lines = file.readlines()
            # for line in lines:
            for line in file:
                line = line.strip()
                if line:  # Non-empty line
                    token, label = line.split("\t")
                    tokens.append(token)
                    labels.append(label)
                else:  # Empty line (sentence separator)
                    if tokens and labels:
                        data.append({tokens_column: tokens, labels_column: labels})
                        tokens, labels = [], []

            # Add the last sentence if the file doesn't end with a blank line
            if tokens and labels:
                data.append({tokens_column: tokens, labels_column: labels})

        if use_pandas:
            data = pd.DataFrame(data)
            logger.info(f"Loaded {len(data)} sentences from {file_path} using pandas.")
        else:
            logger.info(f"Loaded {len(data)} sentences from {file_path} without pandas.")
        return data
    except Exception as e:
        logger.error(f"Error loading CoNLL from {file_path}: {e}")
        raise

def save_conll(data: Union[List[Dict[str, List[str]]], pd.DataFrame], output_path: str, columns=["Tokens", "Labels"], use_pandas: bool = True) -> None:
    """
    Saves data to a CoNLL file.

    Args:
        data (Union[List[Dict[str, List[str]]], pd.DataFrame]): The data to be saved, either as a list of dictionaries or a pandas DataFrame.
        output_path (str): The path where the CoNLL file will be saved.
        columns (list, optional): A list of column names to use for the data. Defaults to ["Tokens", "Labels"]. 
            Note: Only 1 or 2 columns are supported for saving to CoNLL format.
        use_pandas (bool, optional): A flag indicating whether to use pandas for data saving. Defaults to True.

    Raises:
        ValueError: If 'columns' is empty or has more than 2 columns.
        Exception: If an error occurs during the saving process.
    """
    try:
        if not columns or len(columns) > 2:
            logger.warning("'columns' must be provided and can have at most 2 columns. Using default columns: ['Tokens', 'Labels'].")
            columns = ['Tokens', 'Labels']
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if use_pandas:
            if isinstance(data, pd.DataFrame):
                data = data.to_dict("records")
            else:
                data = pd.DataFrame(data).to_dict("records")
        
        with open(output_path, mode='w', encoding='utf-8') as file:
            
            for row in data:
                if len(columns) == 1:
                    for token, label in row[columns[0]]:
                        file.write(f"{token}\t{label}\n")
                    
                elif len(columns) == 2:
                    tokens_column, labels_column = columns
                    for token, label in zip(row[tokens_column], row[labels_column]):
                        file.write(f"{token}\t{label}\n")
                file.write("\n")  # Separate rows/sentences with a blank line

        logger.info(f"Data saved to {output_path} successfully.")
    
    except ValueError as e:
        logger.error(f"Error saving data to {output_path}: {e}")
        raise

def load_data(file_path: str, use_pandas: bool = True) -> Union[List[Dict], Dict, pd.DataFrame]:
    """
    Loads data from a JSON, CSV, CoNLL, or pickle file.

    Args:
        file_path (str): Path to the file to load.
        use_pandas (bool, optional): Whether to use pandas. Defaults to True.

    Returns:
        Union[List[Dict], Dict, pd.DataFrame]: Loaded data.

    Raises:
        ValueError: If the file format is unsupported.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith('.json'):
        return load_json(file_path, use_pandas=use_pandas)
    elif file_path.endswith('.csv'):
        return load_csv(file_path, use_pandas=use_pandas)
    elif file_path.endswith('.pkl'):
        return load_pickle(file_path)
    elif file_path.endswith('.conll'):
        return load_conll(file_path, use_pandas=use_pandas)
    else:
        logger.error("Unsupported file format. Use JSON, CSV, CoNLL, or pickle.")
        raise ValueError("Unsupported file format. Use JSON, CSV, CoNLL, or pickle.")
