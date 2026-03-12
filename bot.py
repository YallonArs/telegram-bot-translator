"""
Main Telegram bot entry point.

Flow for every voice message:
1. Immediately reply → "⏳ Message received…"
2. Download the OGG file to a temp location
3. Edit status → "🎙️ Transcribing audio…"
4. Transcribe with local Whisper (inside a thread-pool thread)
5. Edit status → "🌐 Translating to Russian…"
6. Translate with GoogleTranslator (inside a thread-pool thread)
7. Edit the status message with the final translated text
"""

import asyncio
import logging
import os
import tempfile

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

import config
import transcriber
import translator

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming voice messages end-to-end."""
    message = update.effective_message
    if message is None:
        return

    # 1. Acknowledge instantly
    status_msg = await message.reply_text("⏳ Сообщение получено, обрабатываю…")

    # 2. Download voice file to a temporary file
    voice = message.voice
    tg_file = await context.bot.get_file(voice.file_id)

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name
        await tg_file.download_to_drive(tmp_path)

        # 3. Status: transcribing
        await status_msg.edit_text("🎙️ Транскрибирую аудио…")

        # 4. Transcribe in thread pool (blocking CPU work)
        transcribed_text: str = await asyncio.to_thread(
            transcriber.transcribe, tmp_path
        )

        if not transcribed_text:
            await status_msg.edit_text(
                "❌ Не удалось распознать речь. Попробуй ещё раз."
            )
            return

        # 5. Status: translating
        await status_msg.edit_text("🌐 Перевожу на русский язык…")

        # 6. Translate in thread pool (blocking network I/O)
        translated_text: str = await asyncio.to_thread(
            translator._translate_sync, transcribed_text
        )

        # 7. Final result
        reply = (
            f"✅ *Перевод на русский:*\n\n{translated_text}\n\n"
            f"_Оригинал:_ {transcribed_text}"
        )
        await status_msg.edit_text(reply, parse_mode="Markdown")

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error processing voice message: %s", exc)
        await status_msg.edit_text(
            f"❌ Произошла ошибка: {type(exc).__name__}: {exc}"
        )
    finally:
        # Clean up temp file
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = Application.builder().token(config.BOT_TOKEN).build()

    # Accept voice messages in private chats and groups
    app.add_handler(
        MessageHandler(filters.VOICE, handle_voice)
    )

    logger.info(
        "Bot started. Whisper model: '%s'. Listening for voice messages…",
        config.WHISPER_MODEL,
    )
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
