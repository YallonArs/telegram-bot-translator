"""
Translation module using deep-translator's GoogleTranslator.

No API key required. Source language is auto-detected.
"""

import logging

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

_translator = GoogleTranslator(source="auto", target="ru")


def _translate_sync(text: str) -> str:
    """Blocking translation call — run via asyncio.to_thread()."""
    logger.info("Translating text (%d chars)…", len(text))
    translated: str = _translator.translate(text)
    logger.info("Translation finished: %r", translated[:80])
    return translated
