import os
import sys
import csv
import json
import asyncio
from typing import List, Tuple
from collections import defaultdict

import pandas as pd
# from dotenv import load_dotenv
from telethon.tl.types import Message
# from telethon import TelegramClient
from telethon.sync import TelegramClient
from telethon.errors import FloodWaitError

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

SESSION_FILE = os.path.join('..', 'fetching-E-commerce-data.session')
DATA_PATH = os.path.join('resources', 'data')
OUTPUT_DIR = os.path.join(DATA_PATH, 'raw')
MEDIA_DIR = os.path.join(DATA_PATH, 'photos')

# Ensure the data directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)

async def fetch_messages(client: TelegramClient, channel_username: str, limit: int = 100) -> Tuple[List[dict], List[Message]]:
    """Fetch messages and group media by Group ID from a specified Telegram channel."""
    try:
        messages_data = defaultdict(lambda: {
            "Group ID": None,  # Use grouped_id as the key
            "Message IDs": [],  # List of message IDs in the group
            "Text": None,
            "Message": "",
            "Date": None,
            "Sender ID": None,
            "Media Path": []
        })
        medias = []

        async for message in client.iter_messages(channel_username, limit=limit):

            group_id = message.grouped_id if message.grouped_id else message.id
            message_entry = messages_data[group_id]

            # Update message entry
            message_entry["Group ID"] = group_id
            message_entry["Message IDs"].append(message.id)

            # Only set text once per group
            if message.text and not message_entry["Text"]: 
                message_entry["Text"] = message.text

            # Only set message once per group
            if message.message and not message_entry["Message"]:  
                message_entry["Message"] = message.message
            
            # Only set date once per group
            if not message_entry["Date"]:  
                message_entry["Date"] = message.date.isoformat() if message.date else None
            
            # Only set sender ID once per group
            if not message_entry["Sender ID"]:  
                message_entry["Sender ID"] = message.sender_id

            # Append media if present
            logger.info(f"Multiple photos {str(message.id)} {str(message.grouped_id)}")
            if message.media:
                if(hasattr(message.media, 'photo') or
                    (hasattr(message.media, 'document') and message.media.document.mime_type)):
                    medias.append(message)
                    message_entry["Media Path"].append(None)

        logger.info(f"Fetched {len(messages_data)} message groups from {channel_username}")
        return list(messages_data.values()), medias

    except Exception as e:
        logger.error(f"Error fetching messages from {channel_username}: {e}")
        return [], []

async def scrape_channel_with_photos(medias: List[Message], messages: List[dict], channel_username: str, media_dir: str) -> List[dict]:
    """Fetch and save messages with media concurrently."""
    try:
        channel_media_dir = os.path.join(media_dir, channel_username)
        os.makedirs(channel_media_dir, exist_ok=True)

        if medias:
            media_paths = await download_media(medias, channel_media_dir)

            logger.error(f"Media paths {medias}, {media_paths}")

            # Create a lookup for efficient media path assignment
            media_id_to_path = {media.id: path for media, path in zip(medias, media_paths)}
            
            # Assign media paths to grouped messages
            for message in messages:
                message["Media Path"] = [
                    media_id_to_path.get(media.id)
                    for media in medias
                    if media.id in message["Message IDs"]
                ]

        return messages
    
    except Exception as e:
        logger.error(f"Error processing media for {channel_username}: {e}")
        return messages

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
            writer.writerow(['Group ID', 'Message IDs', 'Message', 'Date', 'Sender ID', 'Media'])  # Header

            for message in messages:
                writer.writerow([
                    message['Group ID'],
                    message['Message IDs'],
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
            writer.writerow(['Channel Username', 'Group ID', 'Message IDs', 'Message', 'Date', 'Sender ID', 'Media'])  # Header

            for channel, messages in all_messages.items():
                for message in messages:
                    writer.writerow([
                        channel,
                        message['Group ID'],
                        message['Message IDs'],
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
        csv_output_path = os.path.join(output_dir, f"{channel_username}_messages_pd.csv")
        df.to_csv(csv_output_path, index=False)
        logger.info(f"Saved {len(messages)} messages to {csv_output_path}")

    if save_json:
        # Save to JSON
        json_output_path = os.path.join(output_dir, f"{channel_username}_messages_pd.json")
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
        csv_output_path = os.path.join(output_dir, "all_channels_messages_pd.csv")
        all_df.to_csv(csv_output_path, index=False)
        logger.info(f"Saved messages to {csv_output_path}")

    if save_json:
        json_output_path = os.path.join(output_dir, "all_channels_messages_pd.json")
        all_df.to_json(json_output_path, orient='records', lines=True, force_ascii=False)
        logger.info(f"Saved all messages to {json_output_path}")

async def process_channel(client, channel, media_dir, limit):
    """Process a single channel: fetch and save messages."""
    try:
        logger.info(f"Fetching messages from: {channel}...")
        
        messages, medias = await fetch_messages(client, channel, limit=limit)
        messages_with_media_path = await scrape_channel_with_photos(medias, messages, channel, media_dir)

        # messages = messages_without_media + messages_with_media_path
        logger.info(f"Processed messages from {channel}.")
        return messages_with_media_path

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
                save_messages(channel, messages, output_dir, save_csv, save_json)
                # save_messages_pd(channel, messages, output_dir, save_csv, save_json)
                logger.info(f"Saved messages from {channel} to {output_dir}")

        except Exception as e:
            logger.error(f"Failed to process channel {channel}: {e}")

    logger.info(f"Processed all channels.")
    return all_messages

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
        client = create_client(session_file=SESSION_FILE)
        try:
            logger.info("Connecting client...")
            await client.connect()

            # Ensure the user is authorized
            logger.info("Authenticating client...")
            if not await client.is_user_authorized():
                await authenticate_client(client)

            # Fetch messages from channels
            all_messages = await process_channels(client, channels, output_dir, media_dir, limit, save_csv=True, save_json=True)

            # Save all messages to a single CSV after processing all channels
            save_all_messages(all_messages, output_dir, save_csv=True, save_json=True)
            # save_all_messages_pd(all_messages, output_dir, save_csv=True, save_json=True)

            # Save session after all done
            await save_session(client, session_file=SESSION_FILE)
            
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
    LIMIT = 500
    run_fetch_process(channels=CHANNELS[10:15], limit=LIMIT)