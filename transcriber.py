"""
Whisper-based transcription module.

The model is loaded once at import time (lazy, on first call) so the bot
starts fast and the heavy model load happens in the background thread.
"""

import logging
from functools import lru_cache

import whisper

import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> whisper.Whisper:
    """Load and cache the Whisper model (called from a thread pool worker)."""
    logger.info("Loading Whisper model '%s'…", config.WHISPER_MODEL)
    model = whisper.load_model(config.WHISPER_MODEL)
    logger.info("Whisper model loaded.")
    return model


def transcribe(audio_path: str) -> str:
    """
    Transcribe an audio file to text.

    Synchronous — designed to be called via ``asyncio.to_thread()``.

    Args:
        audio_path: Path to the audio file (OGG/WAV/MP3 — Whisper handles
                    all of them internally via ffmpeg).

    Returns:
        Transcribed text string.
    """
    model = _get_model()
    logger.info("Transcribing '%s'…", audio_path)
    result = model.transcribe(audio_path, fp16=False)
    text: str = result["text"].strip()
    logger.info("Transcription finished: %r", text[:80])
    return text
