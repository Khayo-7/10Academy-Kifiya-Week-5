import os
import sys
from PIL import Image
import pytesseract

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import load_json

logger = setup_logger("extract")

CONFIG_PATH = os.path.join('..', 'resources', 'configs')
config_file = os.path.join(CONFIG_PATH, 'config.json')

config = load_json(config_file)

TESSERACT_PATH = config['TESSERACT_PATH']

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='amh')
        logger.info(f"Extracted text from image: {image_path}")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from image: {image_path}. Error: {e}")
        return None
