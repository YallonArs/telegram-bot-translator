"""
Whisper-based transcription module (faster-whisper backend).

Supports multiple model sizes simultaneously — each loaded model is cached
in a dict keyed by its name so switching models does not reload everything.
All public functions are synchronous and designed to be called via
``asyncio.to_thread()``.
"""

import logging
from threading import Lock

from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)

_models: dict[str, WhisperModel] = {}
_lock = Lock()

# Sinhala language code
_SINHALA_CODE = "si"


def _get_model(model_name: str) -> WhisperModel:
    """Return a cached WhisperModel, loading it on first use."""
    if model_name not in _models:
        with _lock:
            # Double-checked locking
            if model_name not in _models:
                logger.info("Loading Whisper model '%s'…", model_name)
                _models[model_name] = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                )
                logger.info("Whisper model '%s' loaded.", model_name)
    return _models[model_name]


def list_loaded_models() -> list[str]:
    """Return the names of all currently-cached Whisper models."""
    with _lock:
        return list(_models.keys())


def unload_model(model_name: str) -> bool:
    """
    Remove *model_name* from the in-memory cache.

    Returns True if the model was loaded (and is now removed), False otherwise.
    """
    with _lock:
        if model_name in _models:
            del _models[model_name]
            logger.info("Whisper model '%s' unloaded.", model_name)
            return True
    return False


def transcribe(
    audio_path: str,
    model_name: str = "base",
    language: str | None = None,
) -> str:
    """
    Transcribe an audio file to text.

    Synchronous — call via ``asyncio.to_thread(transcriber.transcribe, path, model)``.

    Args:
        audio_path:  Path to the audio file (OGG/WAV/MP3 — Whisper decodes
                     all formats internally via ffmpeg).
        model_name:  Whisper model size: tiny | base | small | medium | large.
        language:    Target language code. When ``"si"`` (Sinhala), a dedicated
                     local CTranslate2 model is used regardless of *model_name*.

    Returns:
        Transcribed text (stripped).
    """
    # Force the Sinhala-specific model when the target language is Sinhala
    if language == _SINHALA_CODE:
        effective_model = config.SINHALA_MODEL_PATH
        logger.info(
            "Sinhala detected — overriding model to '%s'", effective_model
        )
    else:
        effective_model = model_name

    model = _get_model(effective_model)
    logger.info("Transcribing '%s' with model '%s'…", audio_path, effective_model)

    segments, info = model.transcribe(audio_path, beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments).strip()

    logger.info("Transcription done: %r", text[:80])
    return text
