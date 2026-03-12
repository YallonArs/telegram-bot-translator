"""
Per-user settings store (in-memory).

Each user has:
  model    — Whisper model size  (default: "base")
  language — Translation target  (default: "ru")
"""

from dataclasses import dataclass, field
from typing import ClassVar

# ── available options ────────────────────────────────────────────────────────

WHISPER_MODELS: list[str] = ["base", "small", "medium"]

# (code, display label)
LANGUAGES: list[tuple[str, str]] = [
	("ru", "🇷🇺 Русский"),
	("en", "🇬🇧 English"),
	("de", "🇩🇪 Deutsch"),
	("fr", "🇫🇷 Français"),
	("es", "🇪🇸 Español"),
	("zh-CN", "🇨🇳 中文"),
	("ja", "🇯🇵 日本語"),
	("uk", "🇺🇦 Українська"),
	("tr", "🇹🇷 Türkçe"),
	("ar", "🇸🇦 العربية"),
]

LANGUAGE_LABELS: dict[str, str] = {code: label for code, label in LANGUAGES}

# ── user settings ─────────────────────────────────────────────────────────────


@dataclass
class UserSettings:
	model: str = "base"
	language: str = "ru"
	last_voice_bytes: bytes | None = None
	last_voice_duration: int = 0  # seconds, from Telegram message.voice.duration


# global store: user_id → UserSettings
_store: dict[int, UserSettings] = {}


def get(user_id: int) -> UserSettings:
	if user_id not in _store:
		_store[user_id] = UserSettings()
	return _store[user_id]


def set_model(user_id: int, model: str) -> None:
	get(user_id).model = model


def set_language(user_id: int, language: str) -> None:
	get(user_id).language = language


def set_last_voice(user_id: int, data: bytes, duration: int) -> None:
	s = get(user_id)
	s.last_voice_bytes = data
	s.last_voice_duration = duration
