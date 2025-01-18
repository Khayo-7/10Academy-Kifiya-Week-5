import pandas as pd
import re
import re
import os
import sys
import pandas as pd

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
    amharic_diacritics_map = {
        'ሃ': 'ሀ',
        'ኣ': 'አ' ,
        '':'ሁ', ' ሂ' 'ሓ': 'ሀ', 'ሔ': 'ሄ', 'ሕ': 'ህ', '': 'ሆ',
        'ሐ': 'ሀ', '':'ሁ' ' ሂ', 'ሓ': 'ሀ', 'ሔ': 'ሄ', 'ሕ': 'ህ', '': 'ሆ',
        'ዐ':'አ', 'ዑ': 'ኡ'        
    }
    for diacritic, base_char in amharic_diacritics_map.items():
        text = text.replace(diacritic, base_char)
    logger.info("Normalized Amharic text by removing diacritics.")
    return text.strip()

def tokenize_text(text):
    logger.info("Tokenizing the text.")
    return text.split()

def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def save_labeled_data_to_file(data, file_path, labeled_column):
    with open(file_path, 'w', encoding='utf-8') as f:
        for index, row in data.iterrows():
            f.write(f"{row[labeled_column]}\n\n")

def label_message_utf8_with_birr(message):

    if '\n' in message:
        first_line, remaining_message = message.split('\n', 1)
    else:
        first_line, remaining_message = message, ""
    
    labeled_tokens = []
    
    # Tokenize the first line
    first_line_tokens = re.findall(r'\S+', first_line)
    
    # Label the first token as B-PRODUCT and the rest as I-PRODUCT
    if first_line_tokens:
        labeled_tokens.append(f"{first_line_tokens[0]} B-PRODUCT")  # First token as B-PRODUCT
        for token in first_line_tokens[1:]:
            labeled_tokens.append(f"{token} I-PRODUCT")  # Remaining tokens as I-PRODUCT
    
    # Process the remaining message normally
    if remaining_message:
        lines = remaining_message.split('\n')
        for line in lines:
            tokens = re.findall(r'\S+', line)  # Tokenize each line while considering non-ASCII characters
            
            for token in tokens:
                # Check if token is a price (e.g., 500 ETB, $100, or ብር)
                if re.match(r'^\d{10,}$', token):
                    labeled_tokens.append(f"{token} O")  # Label as O for "other" or outside of any entity
                elif re.match(r'^\d+(\.\d{1,2})?$', token) or 'ETB' in token or 'ዋጋ' in token or '$' in token or 'ብር' in token:
                    labeled_tokens.append(f"{token} I-PRICE")
                # Check if token could be a location (e.g., cities or general location names)
                elif any(loc in token for loc in ['Addis Ababa', 'ለቡ', 'ለቡ መዳህኒዓለም', 'መገናኛ', 'ቦሌ', 'ሜክሲኮ']):
                    labeled_tokens.append(f"{token} I-LOC")
                # Assume other tokens are part of a product name or general text
                else:
                    labeled_tokens.append(f"{token} O")
    
    return "\n".join(labeled_tokens)


def clean_data(data, column):

    data = data.dropna(subset=[column])
    data[column] = data[column].apply(remove_emojis)
    data['Labeled_Message'] = data[column].apply(label_message_utf8_with_birr)
    # save_labeled_data_to_file(data, file_path, labeled_column)


def normalize_text(text):
    """Clean and normalize Amharic text."""
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = text.strip()  # Remove extra spaces
    return text

def preprocess_data(input_file, output_file):
    """Reads raw messages, preprocesses them, and saves the structured dataset."""
    df = pd.read_csv(input_file)
    df["normalized_text"] = df["text"].apply(normalize_text)
    df.dropna(subset=["normalized_text"], inplace=True)  # Drop empty rows
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")
