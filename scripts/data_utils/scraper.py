import os
import sys
import csv
import json
import asyncio
import logging
from typing import List
# from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.sessions import StringSession

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

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
config_file = os.path.join(CONFIG_PATH, 'config.json')

with open(config_file) as f:
    config = json.load(f)

API_ID = config['API_ID']
API_HASH = config['API_HASH']
PHONE_NUMBER = config['PHONE_NUMBER']
# BOT_TOKEN = config['BOT_TOKEN']


SESSION_FILENAME = 'fetching-E-commerce-data'
DATA_PATH = os.path.join('..', '..', 'resources', 'data')
media_dir = os.path.join(DATA_PATH, 'photos')

# Ensure the data directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(media_dir, exist_ok=True)

def create_client(session_file=SESSION_FILENAME):
    """Create and return a Telegram client."""
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_str = f.read().strip()
        return TelegramClient(StringSession(session_str), API_ID, API_HASH)
    else:
        return TelegramClient(StringSession(), API_ID, API_HASH)
        # client = TelegramClient(StringSession(), API_ID, API_HASH).start(bot_token=BOT_TOKEN)

async def save_session(client, session_file=SESSION_FILENAME):
    """Save the current session to a file."""
    with open(session_file, 'w') as f:
        f.write(client.session.save())

def save_messages_to_file(messages, filename='messages.txt'):
    """Save fetched messages to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        for message in messages:
            if message.text:
                file.write(f"{message.date}: {message.text}\n")

async def scrape_channel_with_media(client, channel_username, writer, media_dir):
    entity = await client.get_entity(channel_username)
    channel_title = entity.title  # Extract the channel's title
    async for message in client.iter_messages(entity, limit=10000):
        media_path = None
        if message.media and hasattr(message.media, 'photo'):
            filename = f"{channel_username}_{message.id}.jpg"
            media_path = os.path.join(media_dir, filename)
            await client.download_media(message.media, media_path)
        
        # Write the channel title along with other data
        writer.writerow([channel_title, channel_username, message.id, message.message, message.date, media_path])

async def save_messages_to_file(client, messages, channels, filename='telegram_data.csv'):

    # Open the CSV file and prepare the writer
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])

        for channel in channels:
            try:
                await scrape_channel_with_media(client, channel, writer, media_dir)
                print(f"Scraped data from {channel}")
            except Exception as e:
                print(f"Failed to process channel {channel}: {e}")

async def fetch_messages(client, channel_name, limit=100):
    """Fetch messages from a specified Telegram channel."""
    channel = await client.get_entity(channel_name)
    messages = await client.get_messages(channel, limit=limit)
    return messages

async def authenticate_client(client: TelegramClient):
    """Authenticate the Telegram client."""
    if not await client.is_user_authorized():
        logger.info("You need to log in.")
        try:
            await client.send_code_request(PHONE_NUMBER)
            code = input("Enter the code you received: ")
            await client.sign_in(PHONE_NUMBER, code)
        except FloodWaitError as e:
            logger.warning(f"FloodWaitError for authenticating client: Wait for {e.seconds} seconds before retrying.")
            # logger.warning(f"FloodWaitError for authenticating client: Waiting {e.seconds} seconds.")
            # await asyncio.sleep(e.seconds)
        except Exception as e:
            logger.error(f"Error during login: {e}")
            raise RuntimeError(f"Error during login: {e}")

async def process_channel(client, channel, output_path, limit):
    """Process a single channel: fetch and save messages."""
    try:
        logger.info(f"Fetching messages from: {channel}...")
        messages = await fetch_messages(client, channel, limit)
        filename = os.path.join(output_path, f"{channel}_messages.txt")
        save_messages_to_file(messages, filename)
        logger.info(f"Saved messages from {channel} to {filename}")

    except FloodWaitError as e:
        logger.warning(f"FloodWaitError for {channel}: Wait for {e.seconds} seconds before retrying.")
        # logger.warning(f"FloodWaitError for {channel}: Waiting {e.seconds} seconds.")
        # await asyncio.sleep(e.seconds)

    except Exception as e:
        logger.error(f"Error processing channel {channel}: {e}")

def sync(func):
    """Decorator to run async functions synchronously."""
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

@sync
def main(channels, output_path=DATA_PATH, limit=100):
    """Main function to orchestrate fetching and saving messages."""
    async def process():
        # async with create_client() as client:
        client = create_client()
        try:
            logger.info("Connecting client...")
            await client.connect()

            # Ensure the user is authorized
            logger.info("Authenticating client...")
            await authenticate_client(client)

            messages=[]
            await save_messages_to_file(client, messages, channels)

            # Fetch messages from channels
            for channel in channels:
                await process_channel(client, channel, output_path, limit)

            # Save session after all done
            await save_session(client)
            
        finally:
            # Always disconnect the client
            logger.info("Disconnecting client...")
            await client.disconnect()

    # Run the asynchronous process
    return process()

if __name__ == "__main__":
    
    # List of channels to scrape
    CHANNELS = [
        "ZemenExpress", "nevacomputer", "meneshayeofficial", "ethio_brand_collection", "Leyueqa",
        "sinayelj", "Shewabrand", "helloomarketethiopia", "modernshoppingcenter", "qnashcom",
        "Fashiontera", "kuruwear", "gebeyaadama", "MerttEka", "forfreemarket", "classybrands",
        "marakibrand", "aradabrand2", "marakisat2", "belaclassic", "AwasMart", "qnashcom"
    ]

    # Run only for selected channels
    main(channels=CHANNELS[11:13])