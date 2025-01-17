import re
import os
import sys

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("cleaning")
def clean_text(text):
    if not text:
        logger.info("Input text is empty.")
        return ""
    
    # Remove emojis and special characters
    text = re.sub(r'[^\w\s]', '', text)
    logger.info("Removed emojis and special characters from the text.")
    # Normalize Amharic text
    text = text.replace('ሃ', 'ሀ').replace('ሐ', 'ሀ') # Remove diacritics
    logger.info("Normalized Amharic text by removing diacritics.")
    return text.strip()

def tokenize_text(text):
    logger.info("Tokenizing the text.")
    return text.split()
