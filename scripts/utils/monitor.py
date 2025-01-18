import os
import sys
import json
import asyncio
# from dotenv import load_dotenv
from telethon import TelegramClient, events

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import load_json
from scripts.utils.database import create_table, save_to_database
from scripts.utils.telegram_client import create_client, download_media

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)
logger = setup_logger("scraper")

# Load environment variables
# load_dotenv()

# API_ID = os.getenv("API_ID")
# API_HASH = os.getenv("API_HASH")
# PHONE_NUMBER = os.getenv("PHONE_NUMBER")
# BOT_TOKEN = os.getenv("BOT_TOKEN")

CONFIG_PATH = os.path.join('..', 'resources', 'configs')
MEDIA_DIR = os.path.join('..', 'resources', 'downloads')
SESSION_FILENAME = 'monitoring-E-commerce-channels.session'
config_filepath = os.path.join(CONFIG_PATH, 'config.json')
channels_filepath = os.path.join(CONFIG_PATH, 'channels.json')

config = load_json(config_filepath)

API_ID = config['API_ID']
API_HASH = config['API_HASH']
PHONE_NUMBER = config['PHONE_NUMBER']
# BOT_TOKEN = config['BOT_TOKEN']

# Define a semaphore for limiting concurrent downloads
semaphore = asyncio.Semaphore(5)

# Process incoming messages
async def process_message(event, media_dir):
    """Process incoming Telegram messages, downloading media if necessary and saving to the database."""
    
    channel_username = event.chat.username
    message_id = event.message.id
    text = event.message.message or ''
    media_path = None
    media_type = None

    try:
        # Call download_media with a single message wrapped in a list
        downloaded_paths = await download_media([event.message], media_dir)
        logger.info(f"Downloaded paths: {downloaded_paths}")
        # Save to the database (assuming this function is defined elsewhere)
        await save_to_database(channel_username, message_id, text, media_path, media_type)
    except Exception as e:
        logger.error(f"Error processing message from {channel_username}: {e}")

async def main(channels_filepath, media_dir):
    """Main function to set up the client and listen for new messages."""

    # Load channel usernames from a JSON file
    channel_usernames = load_json(channels_filepath)

    # Create PostgreSQL table if it doesn't exist
    create_table()

    client = create_client(session_file=SESSION_FILENAME)

    # Start listening for new messages in the specified channel
    @client.on(events.NewMessage(chats=channel_usernames))
    async def message_handler(event):
        await process_message(event, media_dir)

    await client.start()
    logger.info("Client is running...")
    await client.run_until_disconnected()

if __name__ == '__main__':

    asyncio.run(main(channels_filepath, MEDIA_DIR))