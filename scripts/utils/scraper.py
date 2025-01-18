import os
import sys
import csv
import json
import getpass
import asyncio
import logging
from typing import List
# from dotenv import load_dotenv
import pandas as pd
from telethon import TelegramClient
from telethon.errors import FloodWaitError, SessionPasswordNeededError
from telethon.sessions import StringSession

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import load_json
from scripts.utils.telegram_client import download_media, create_client, save_session, authenticate_client

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
config_filepath = os.path.join(CONFIG_PATH, 'config.json')
config = load_json(config_filepath)

API_ID = config['API_ID']
API_HASH = config['API_HASH']
PHONE_NUMBER = config['PHONE_NUMBER']
# BOT_TOKEN = config['BOT_TOKEN']

SESSION_FILENAME = 'fetching-E-commerce-data.session'
DATA_PATH = os.path.join('..', 'resources', 'data')
OUTPUT_DIR = os.path.join(DATA_PATH, 'raw')
MEDIA_DIR = os.path.join(DATA_PATH, 'photos')

# Ensure the data directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)

async def fetch_messages(client, channel_username, limit=100):
    """Fetch messages from a specified Telegram channel."""
    try:
        # channel = await client.get_entity(channel_username)
        # messages = await client.get_messages(channel, limit=limit)
        # async for message in client.iter_messages(channel, limit=limit):

        messages_with_photos = []
        messages_without_media = []
        medias = []

        async for message in client.iter_messages(channel_username, limit=limit):
            message_data = {
                "Message ID": message.id,
                "Text": message.text,
                "Message": message.message or "",
                "Date": message.date.isoformat() if message.date else None,
                "Sender ID": message.sender_id,
                "Media Path": None
            }
            if message.media and hasattr(message.media, 'photo'):
                messages_with_photos.append(message_data)
                medias.append(message)
            else:
                messages_without_media.append(message_data)

        logger.info(f"Fetched {len(messages_without_media)+len(messages_without_media)} messages from {channel_username}.")         
        return messages_with_photos, messages_without_media, medias
    
    except Exception as e:
        logger.error(f"Error fetching messages from {channel_username}: {e}")
        return []

async def scrape_channel_with_photos(medias, messages, channel_username, media_dir):
    """Fetch and save messages with media concurrently."""
    try:

        channel_media_dir = os.path.join(media_dir, channel_username)
        os.makedirs(channel_media_dir, exist_ok=True)

        if medias:
            media_paths = await download_media(medias, channel_media_dir)

            for message, media_path in zip(messages, media_paths):
                message['Media Path'] = media_path

        return messages
    
    except Exception as e:
        logger.error(f"Error processing media for {channel_username}: {e}")

async def process_channel(client, channel, media_dir, limit):
    """Process a single channel: fetch and save messages."""
    try:
        logger.info(f"Fetching messages from: {channel}...")
        
        messages_with_photos, messages_without_media, medias = await fetch_messages(client, channel, limit=limit)
        messages_with_media_path = await scrape_channel_with_photos(medias, messages_with_photos, channel, media_dir)

        messages = messages_without_media + messages_with_media_path
        logger.info(f"Processed messages from {channel}.")
        return messages

    except FloodWaitError as e:
        logger.warning(f"FloodWaitError for {channel}: Wait for {e.seconds} seconds before retrying.")
        # logger.warning(f"FloodWaitError for {channel}: Waiting {e.seconds} seconds.")
        # await asyncio.sleep(e.seconds)

    except Exception as e:
        logger.error(f"Error processing channel {channel}: {e}")
        
async def process_channels(client, channels, output_dir, media_dir, limit=100, save_csv=True, save_json=False):
    """Process messages from multiple channels, handling text and media."""
    all_messages = {}
    
    for channel in channels:
        logger.info(f"Processing channel: {channel}...")
        try:
            messages = await process_channel(client, channel, media_dir, limit)
            
            all_messages[channel] = messages
            if save_csv or save_json:
                save_messages(channel, messages, output_dir, save_csv, save_json)  # Save based on options
                logger.info(f"Saved messages from {channel} to {output_dir}")

        except Exception as e:
            logger.error(f"Failed to process channel {channel}: {e}")

    logger.info(f"Processed all channels.")
    return all_messages

def save_messages(channel_username, messages, output_dir, save_csv=True, save_json=False):
    """Save fetched messages into CSV and/or JSON files based on options."""
    if not messages:
        logger.warning(f"No messages to save for {channel_username}.")
        return

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    if save_csv:
        # Save to CSV
        csv_output_path = os.path.join(output_dir, f"{channel_username}_messages.csv")
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Message ID', 'Message', 'Date', 'Sender ID', 'Media'])  # Header

            for message in messages:
                writer.writerow([
                    message['Message ID'],
                    message['Message'],
                    message.get('Date', ''),
                    message['Sender ID'],
                    message.get('Media Path', '')
                ])

        logger.info(f"Saved {len(messages)} messages to {csv_output_path}")

    if save_json:
        # Save to JSON
        json_output_path = os.path.join(output_dir, f"{channel_username}_messages.json")
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(messages, json_file, ensure_ascii=False, indent=4)

        logger.info(f"Saved {len(messages)} messages to {json_output_path}")

def save_all_messages(all_messages, output_dir, save_csv=True, save_json=False):
    """Save all fetched messages from multiple channels into a single CSV and/or JSON file."""
    if not all_messages:
        logger.warning("No messages to save for any channel.")
        return

    os.makedirs(output_dir, exist_ok=True)

    if save_csv:
        csv_output_path = os.path.join(output_dir, "all_channels_messages.csv")
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Channel Username', 'Message ID', 'Message', 'Date', 'Sender ID', 'Media'])  # Header

            for channel, messages in all_messages.items():
                for message in messages:
                    writer.writerow([
                        channel,
                        message['Message ID'],
                        message['Message'],
                        message.get('Date', ''),
                        message['Sender ID'],
                        message.get('Media Path', '')
                    ])

        logger.info(f"Saved all messages to {csv_output_path}")

    if save_json:
        json_output_path = os.path.join(output_dir, "all_channels_messages.json")
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(all_messages, json_file, ensure_ascii=False, indent=4)

        logger.info(f"Saved all messages to {json_output_path}")


def save_messages_pd(channel_username, messages, output_dir, save_csv=True, save_json=False):
    """Save fetched messages into CSV and/or JSON files based on options."""
    if not messages:
        logger.warning(f"No messages to save for {channel_username}.")
        return

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert messages to DataFrame
    df = pd.DataFrame(messages)

    if save_csv:
        # Save to CSV
        csv_output_path = os.path.join(output_dir, f"{channel_username}_messages.csv")
        df.to_csv(csv_output_path, index=False)
        logger.info(f"Saved {len(messages)} messages to {csv_output_path}")

    if save_json:
        # Save to JSON
        json_output_path = os.path.join(output_dir, f"{channel_username}_messages.json")
        df.to_json(json_output_path, orient='records', lines=True, force_ascii=False)
        logger.info(f"Saved {len(messages)} messages to {json_output_path}")

def save_all_messages_pd(all_messages, output_dir, save_csv=True, save_json=False):
    """Save all fetched messages from multiple channels into a single CSV and/or JSON file."""
    if not all_messages:
        logger.warning("No messages to save for any channel.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Convert all messages to DataFrame
    all_df = pd.concat([pd.DataFrame(messages).assign(Channel=channel) for channel, messages in all_messages.items()])

    if save_csv:
        csv_output_path = os.path.join(output_dir, "all_channels_messages.csv")
        all_df.to_csv(csv_output_path, index=False)
        logger.info(f"Saved messages to {csv_output_path}")

    if save_json:
        json_output_path = os.path.join(output_dir, "all_channels_messages.json")
        all_df.to_json(json_output_path, orient='records', lines=True, force_ascii=False)
        logger.info(f"Saved all messages to {json_output_path}")


def sync(func):
    """Decorator to run async functions synchronously."""
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

@sync
def run_fetch_process(channels, output_dir=OUTPUT_DIR, media_dir=MEDIA_DIR, limit=100):
    """Main function to orchestrate fetching and saving messages."""
    async def main():
        # async with create_client() as client:
        client = create_client(session_file=SESSION_FILENAME)
        try:
            logger.info("Connecting client...")
            await client.connect()

            # Ensure the user is authorized
            logger.info("Authenticating client...")
            if not await client.is_user_authorized():
                await authenticate_client(client)

            # Fetch messages from channels
            all_messages = await process_channels(client, channels, output_dir, media_dir, limit)

            # Save all messages to a single CSV after processing all channels
            save_all_messages(all_messages, output_dir, save_csv=True, save_json=True)

            # Save session after all done
            await save_session(client, session_file=SESSION_FILENAME)
            
        finally:
            # Always disconnect the client
            logger.info("Disconnecting client...")
            await client.disconnect()

    # Run the asynchronous process
    return main()

if __name__ == "__main__":
    
    # List of channels to scrape
    CHANNELS = [
        "ZemenExpress", "nevacomputer", "meneshayeofficial", "ethio_brand_collection", "Leyueqa",
        "sinayelj", "Shewabrand", "helloomarketethiopia", "modernshoppingcenter", "qnashcom",
        "Fashiontera", "kuruwear", "gebeyaadama", "MerttEka", "forfreemarket", "classybrands",
        "marakibrand", "aradabrand2", "marakisat2", "belaclassic", "AwasMart", "qnashcom"
    ]
    
    # Run only for selected channels
    LIMIT = 1
    run_fetch_process(channels=CHANNELS[11:12], limit=LIMIT)