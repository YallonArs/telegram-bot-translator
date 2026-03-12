"""
Whisper-based transcription module.

Supports multiple model sizes simultaneously — each loaded model is cached
in a dict keyed by its name so switching models does not reload everything.
All public functions are synchronous and designed to be called via
``asyncio.to_thread()``.
"""

import logging
from threading import Lock

import whisper

logger = logging.getLogger(__name__)

_models: dict[str, whisper.Whisper] = {}
_lock = Lock()


def _get_model(model_name: str) -> whisper.Whisper:
    """Return a cached Whisper model, loading it on first use."""
    if model_name not in _models:
        with _lock:
            # Double-checked locking
            if model_name not in _models:
                logger.info("Loading Whisper model '%s'…", model_name)
                _models[model_name] = whisper.load_model(model_name)
                logger.info("Whisper model '%s' loaded.", model_name)
    return _models[model_name]


def transcribe(audio_path: str, model_name: str = "base") -> str:
    """
    Transcribe an audio file to text.

    Synchronous — call via ``asyncio.to_thread(transcriber.transcribe, path, model)``.

    Args:
        audio_path:  Path to the audio file (OGG/WAV/MP3 — Whisper decodes
                     all formats internally via ffmpeg).
        model_name:  Whisper model size: tiny | base | small | medium | large.

    Returns:
        Transcribed text (stripped).
    """
    model = _get_model(model_name)
    logger.info("Transcribing '%s' with model '%s'…", audio_path, model_name)
    result = model.transcribe(audio_path, fp16=False)
    text: str = result["text"].strip()
    logger.info("Transcription done: %r", text[:80])
    return text
