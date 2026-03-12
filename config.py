import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
