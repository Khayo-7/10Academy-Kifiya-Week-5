import os
import csv
import json
import pandas as pd
from scripts.utils.logger import setup_logger

# Setup logger for data_loader
logger = setup_logger("data_loader")

def load_csv(file_path: str, sep=',') -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        sep (str, optional): Separator to use. Defaults to ','.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        logger.info(f"Loading data from file ...")
        data = pd.read_csv(file_path, sep=sep)
        # dataframe = pd.read_csv(file_path, sep='|')
        # dataframe = pd.read_csv(file_path, sep='\t')
        logger.info(f"Loaded data from {file_path}, shape: {data.shape}")

        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def save_csv(dataframe: pd.DataFrame, output_path: str) -> None:
    """
    Saves the dataframe to a CSV file.

    Args:
        dataframe (pd.DataFrame): The dataframe to save.
        output_path (str): The path to save the dataframe.

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path} successfully.")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")

def load_json(file_path):
    """
    Load a JSON file into a Python object.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        object: Loaded JSON object.
    """
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None

def save_json(dataframe: pd.DataFrame, output_path: str) -> None:
    """
    Saves the dataframe to a JSON file.

    Args:
        dataframe (pd.DataFrame): The dataframe to save.
        output_path (str): The path to save the dataframe.

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataframe.to_json(output_path, index=False)
        logger.info(f"Data saved to {output_path} successfully.")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")

def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loads data from a JSON or CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the file to load.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith('.json'):
        try:
            return pd.read_json(file_path, lines=True)
        except Exception as e:
            logger.error(f"Error loading JSON from {file_path}: {e}")
            raise
    elif file_path.endswith('.csv'):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading CSV from {file_path}: {e}")
            raise
    else:
        logger.error("Unsupported file format. Use JSON or CSV.")
        raise ValueError("Unsupported file format. Use JSON or CSV.")

def save_dataframe(data: pd.DataFrame, filename: str, output_dir: str, save_in_csv: bool = True, save_in_json: bool = False) -> None:
    """
    Saves the dataframe to a CSV and/or JSON file.

    Args:
        data (pd.DataFrame): The dataframe to save.
        filename (str): The base name of the file to save.
        output_dir (str): The directory to save the file.
        save_in_csv (bool, optional): If True, saves the dataframe to a CSV file. Defaults to True.
        save_in_json (bool, optional): If True, saves the dataframe to a JSON file. Defaults to False.
    """
    if not data:
        logger.warning("No data to save for dataframe.")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    if save_in_csv:
        csv_output_path = os.path.join(output_dir, f"{filename}.csv")
        try:
            data.to_csv(csv_output_path, index=False)
            logger.info(f"Data saved to {csv_output_path} successfully.")
        except Exception as e:
            logger.error(f"Error saving data to {csv_output_path}: {e}")

    if save_in_json:
        json_output_path = os.path.join(output_dir, f"{filename}.json")
        try:
            data.to_json(json_output_path, orient='records', lines=True, force_ascii=False)
            logger.info(f"Data saved to {json_output_path} successfully.")
        except Exception as e:
            logger.error(f"Error saving data to {json_output_path}: {e}")

def load_data(file_path: str):
    """
    Loads data from a JSON or CSV file into a Python object.

    Args:
        file_path (str): Path to the file to load.

    Returns:
        object: Loaded data.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith('.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error loading JSON from {file_path}: {e}")
            raise
    elif file_path.endswith('.csv'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                return list(csv_reader)
        except Exception as e:
            logger.error(f"Error loading CSV from {file_path}: {e}")
            raise
    else:
        logger.error("Unsupported file format. Use JSON or CSV.")
        raise ValueError("Unsupported file format. Use JSON or CSV.")
    
def save_data(data, output_dir, filename, save_in_csv=True, save_in_json=False):
    """
    Saves the data to a CSV and/or JSON file.

    Args:
        data (list or pd.DataFrame): The data to save.
        output_dir (str): The directory to save the file.
        filename (str): The base name of the file to save.
        save_in_csv (bool, optional): If True, saves the data to a CSV file. Defaults to True.
        save_in_json (bool, optional): If True, saves the data to a JSON file. Defaults to False.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def save_to_file(file_path, content):
        """
        Saves content to a file.

        Args:
            file_path (str): The path to the file to save.
            content (str): The content to write to the file.
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Data saved to {file_path} successfully.")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise
    
    if save_in_csv:
        csv_output_path = os.path.join(output_dir, f"{filename}.csv")
        csv_content = "\n".join([str(row) for row in data])
        save_to_file(csv_output_path, csv_content)
        
    if save_in_json:
        json_output_path = os.path.join(output_dir, f"{filename}.json")
        json_content = "\n\n".join([str(row) for row in data])
        save_to_file(json_output_path, json_content)
