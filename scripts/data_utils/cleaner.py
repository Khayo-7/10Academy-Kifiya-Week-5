import re
import os
import sys
from typing import Dict

# Setup logger for cleaning operations
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("cleaning")

# ==========================================
# Helper Functions for Cleaning Operations
# ==========================================

def normalize_amharic_text(text: str, diacritics_map: Dict[str, str]) -> str:
    """
    Replaces Amharic diacritics with their base forms based on the given map.
    """
    if not isinstance(text, str):
        logger.warning("Input text is not a string. Skipping normalization.")
        return text

    for diacritic, base_char in diacritics_map.items():
        text = text.replace(diacritic, base_char)
    
    logger.debug("Normalized Amharic diacritics.")
    return text

def remove_non_amharic_characters(text: str) -> str:
    """
    Removes characters that are not part of the Amharic Unicode block or spaces.
    """
    # pattern = re.compile(r'[^\u1200-\u137F\s]') # Retains Amharic script only
    pattern = re.compile(r'[^\u1200-\u137F0-9\s]')  # Retains Amharic script and numbers
    result = pattern.sub('', text)
    logger.debug("Removed non-Amharic characters.")
    return result

def remove_punctuation(text: str) -> str:
    """
    Removes Amharic punctuation and replaces it with a space.
    """
    pattern = re.compile(r'[፡።፣፤፥፦፧፨]+')
    result = pattern.sub(' ', text)
    logger.debug("Removed Amharic punctuation.")
    return result

def remove_emojis(text: str) -> str:
    """
    Removes emojis from the text.
    """
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    result = emoji_pattern.sub('', text)
    logger.debug("Removed emojis.")
    return result

def remove_repeated_characters(text: str) -> str:
    """
    Collapses consecutive repeated characters into a single instance.
    """
    result = re.sub(r'(.)\1+', r'\1', text)
    logger.debug("Removed repeated characters.")
    return result

def remove_numbers(text: str) -> str:
    """
    Removes numeric characters from the text.
    """
    result = re.sub(r'\d+', '', text)
    logger.debug("Removed numbers.")
    return result

def remove_urls(text: str) -> str:
    """
    Removes URLs from the text.
    """
    result = re.sub(r'http\S+|www\S+', '', text)
    logger.debug("Removed URLs.")
    return result

def normalize_spaces(text: str) -> str:
    """
    Normalize spaces in the text.
    """
    # Normalize Multiple whitespace characters and trim
    result = ' '.join(text.split()).strip()
    # result = re.sub(r'\s+', ' ', text).strip()

    logger.debug("Normalize spaces.")
    return result

def clean_text_pipeline(text: str) -> str:
    """
    Orchestrates the text cleaning process by applying multiple cleaning functions.
    """
    if not text:
        logger.info("Received empty input text.")
        return ""

    # Define the Amharic diacritics map
    amharic_diacritics_map = {
        'ኀ': 'ሀ', 'ኁ': 'ሁ', 'ኂ': 'ሂ', 'ኃ': 'ሀ', 'ኄ': 'ሄ', 'ኅ': 'ህ', 'ኆ': 'ሆ',
        'ሐ': 'ሀ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሀ', 'ሔ': 'ሄ', 'ሕ': 'ህ', 'ሖ': 'ሆ',
        'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ',
        'ዐ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዓ': 'አ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ', 'ኣ': 'አ'
    }

    # Apply cleaning steps
    text = normalize_amharic_text(text, amharic_diacritics_map)
    text = remove_non_amharic_characters(text)
    text = remove_punctuation(text)
    text = remove_emojis(text)
    text = remove_repeated_characters(text)
    text = remove_urls(text)
    # text = remove_numbers(text)
    text = normalize_spaces(text)

    logger.debug("Final text cleaning completed.")
    return text
