"""
Translation module using deep-translator's GoogleTranslator.

No API key required. Source language is auto-detected.
Supports dynamic target language per call.
"""

import logging

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)


def translate(text: str, target: str = "ru") -> str:
    """
    Translate *text* into *target* language.

    Synchronous — call via ``asyncio.to_thread(translator.translate, text, lang)``.

    Args:
        text:   Source text (language auto-detected).
        target: BCP-47 language code, e.g. "ru", "en", "de", "zh-CN".

    Returns:
        Translated string.
    """
    logger.info("Translating %d chars → '%s'…", len(text), target)
    translated: str = GoogleTranslator(source="auto", target=target).translate(text)
    logger.info("Translation done: %r", translated[:80])
    return translated
