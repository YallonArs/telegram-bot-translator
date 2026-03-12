import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

# Comma-separated list of chat IDs that have full access.
# Example in .env:  ALLOWED_CHAT_IDS=123456789,987654321
_raw_ids = os.getenv("ALLOWED_CHAT_IDS", "")
ALLOWED_CHAT_IDS: frozenset[int] = frozenset(
    int(x.strip()) for x in _raw_ids.split(",") if x.strip()
)

# Single admin chat ID with access to the /admin panel.
# Example in .env:  ADMIN_CHAT_ID=123456789
_admin_raw = os.getenv("ADMIN_CHAT_ID", "").strip()
ADMIN_CHAT_ID: int | None = int(_admin_raw) if _admin_raw else None
