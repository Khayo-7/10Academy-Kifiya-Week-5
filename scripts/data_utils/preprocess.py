import os
import sys
import pandas as pd

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.cleaner import clean_text_pipeline
from scripts.data_utils.loaders import save_csv, save_json

logger = setup_logger("preprocess")

# ==========================================
# Main Data Preprocessing Function
# ==========================================

def clean_data(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Cleans a specific column in the DataFrame by applying the text cleaning pipeline.
    """
    if column not in data.columns:
        logger.error(f"Column '{column}' not found in the DataFrame.")
        raise ValueError(f"Column '{column}' does not exist.")

    # Drop rows with missing values in the specified column
    data = data.dropna(subset=[column])
    logger.info(f"Dropped rows with NaN values in column '{column}'.")

    # Apply cleaning pipeline
    data[column] = data[column].apply(clean_text_pipeline)    

    logger.info(f"Text cleaning applied to column '{column}'.")
    
    return data

def split_and_explode(data: pd.DataFrame, column: str, delimeter: str = '.') -> pd.DataFrame:
    """
    Splits the text in the specified column by given delimiter and then explodes the DataFrame.
    """
    data[column] = data[column].apply(lambda x: x.split(delimeter))
    data = data.explode(column).reset_index(drop=True)
    return data

def preprocess_data(data: pd.DataFrame, column: str, filename: str, output_dir: str, explode: bool = False, save_in_csv: bool = True, save_in_json: bool = False) -> pd.DataFrame:
    """
    Loads, preprocesses, and saves cleaned data.
    """
    logger.info("Preprocessing text data...")
    cleaned_data = clean_data(data, column)

    logger.info("Removing empty...")
    preprocessed_data = cleaned_data.dropna(subset=[column])

    if explode:
        preprocessed_data = split_and_explode(preprocessed_data, column)

    preprocessed_data = preprocessed_data.reset_index(drop=True)

    logger.info("Saving preprocessed data...")
    output_file = os.path.join(output_dir, filename)
    if save_in_csv:
        save_csv(preprocessed_data, output_file + ".csv")
    if save_in_json:
        save_json(preprocessed_data, output_file + ".json")

    # save_dataframe(preprocessed_data, filename, output_dir, save_in_csv, save_in_json)

    logger.info(f"Preprocessed data saved to {output_dir}")


    return preprocessed_data
