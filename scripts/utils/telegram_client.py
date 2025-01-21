import os
import sys
import getpass
import asyncio
from typing import List

# from dotenv import load_dotenv
from telethon.tl.types import Message
# from telethon import TelegramClient
from telethon.sync import TelegramClient
from telethon.sessions import StringSession
from telethon.errors import FloodWaitError, SessionPasswordNeededError


# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import load_json

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(media)s")
# logger = logging.getLogger(__name__)
logger = setup_logger("scraper")

# Load environment variables
# load_dotenv()

# API_ID = os.getenv("API_ID")
# API_HASH = os.getenv("API_HASH")
# PHONE_NUMBER = os.getenv("PHONE_NUMBER")
# TELEGRAM_SCRAPER_SEMAPHORE = os.getenv("TELEGRAM_SCRAPER_SEMAPHORE")
# BOT_TOKEN = os.getenv("BOT_TOKEN")

CONFIG_PATH = os.path.join('..', 'resources', 'configs')
config_filepath = os.path.join(CONFIG_PATH, 'config.json')
config = load_json(config_filepath)

API_ID = config['API_ID']
API_HASH = config['API_HASH']
PHONE_NUMBER = config['PHONE_NUMBER']
TELEGRAM_SCRAPER_SEMAPHORE = config['TELEGRAM_SCRAPER_SEMAPHORE']
# BOT_TOKEN = config['BOT_TOKEN']

# Define a semaphore for limiting concurrent downloads
semaphore_limit = int(TELEGRAM_SCRAPER_SEMAPHORE)
semaphore = asyncio.Semaphore(semaphore_limit)

def create_client(session_file="telegram.session"):
    """Create and return a Telegram client."""
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_str = f.read().strip()
        return TelegramClient(StringSession(session_str), API_ID, API_HASH)
    else:
        return TelegramClient(StringSession(), API_ID, API_HASH)
        # client = TelegramClient(StringSession(), API_ID, API_HASH).start(bot_token=BOT_TOKEN)
    
async def save_session(client, session_file="telegram.session"):
    """Save the current session to a file."""
    with open(session_file, 'w') as f:
        f.write(client.session.save())

async def authenticate_client(client):
    """Authenticate the Telegram client."""
    # await client.start()
    if not await client.is_user_authorized():
        logger.info("You need to log in.")
        try:
            await client.send_code_request(PHONE_NUMBER)
            code = input("Enter the code you received: ")
            await client.sign_in(PHONE_NUMBER, code)
        except SessionPasswordNeededError:
            password = getpass.getpass("Enter your 2AF password: ")
            await client.sign_in(password=password)
        except FloodWaitError as e:
            logger.warning(f"FloodWaitError when authenticating client: Wait for {e.seconds} seconds before retrying.")
            # logger.warning(f"FloodWaitError: Waiting {e.seconds} seconds.")
            # await asyncio.sleep(e.seconds)
        except Exception as e:
            logger.error(f"Error during login: {e}")
            raise RuntimeError(f"Error during login: {e}")

def download_concurrently(func):
    """Decorator to download concurrently."""
    async def wrapper(*args, **kwargs):
        tasks = await func(*args, **kwargs)
        return await asyncio.gather(*tasks)
    return wrapper

@download_concurrently
async def download_media(medias: List[Message], media_dir: str) -> List[str]:
    """Download media from messages while respecting the semaphore limit."""

    # Ensure the media directory exists
    os.makedirs(media_dir, exist_ok=True)
    
    async def download(media):
        # Determine the file path based on the media type
        try:
            file_ext = (
                "jpg" if hasattr(media.media, 'photo')
                else media.media.document.mime_type.split('/')[-1]
                if hasattr(media.media, 'document') and media.media.document.mime_type 
                else "bin"
            )
            filename = f"{media.id}.{file_ext}"
            media_path = os.path.join(media_dir, filename)

            logger.info(f"Preparing to download: {media_path}")
            async with semaphore:
                media_path = await media.download_media(media_path)
                logger.info(f"Downloaded: {media_path}")
                return media_path

        except Exception as e:
            logger.error(f"Failed to download media for message {media.id}: {e}")

    tasks = [download(media) for media in medias]
    return tasks
