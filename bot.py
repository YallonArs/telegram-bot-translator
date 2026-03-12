"""
Telegram bot — aiogram 3.x rewrite.

Features
--------
* Accepts voice messages in private chats and groups.
* Live status editing: received → transcribing → translating → result.
* /settings command: inline keyboard to switch Whisper model and target language.
* /benchmark command: runs all Whisper models on last received voice, reports timing.
* Per-user settings stored in memory (settings.py).
* Fully async: blocking Whisper inference and HTTP calls run in thread pool.
"""

import asyncio
import logging
import os
import tempfile
import time

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import (
	CallbackQuery,
	InlineKeyboardButton,
	InlineKeyboardMarkup,
	Message,
)

import config
import settings as user_settings
import transcriber
import translator

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
	format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
	level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── aiogram setup ─────────────────────────────────────────────────────────────

bot = Bot(
	token=config.BOT_TOKEN,
	default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
)
dp = Dispatcher()

# ── helpers ───────────────────────────────────────────────────────────────────


def _settings_keyboard(user_id: int) -> InlineKeyboardMarkup:
	"""Build the settings inline keyboard for *user_id*."""
	s = user_settings.get(user_id)

	# ── Model row ──
	model_buttons: list[InlineKeyboardButton] = []
	for m in user_settings.WHISPER_MODELS:
		label = f"✅ {m}" if m == s.model else m
		model_buttons.append(InlineKeyboardButton(text=label, callback_data=f"model:{m}"))

	# ── Language rows (2 per row) ──
	lang_buttons: list[list[InlineKeyboardButton]] = []
	row: list[InlineKeyboardButton] = []
	for code, label in user_settings.LANGUAGES:
		mark = "✅ " if code == s.language else ""
		btn = InlineKeyboardButton(text=f"{mark}{label}", callback_data=f"lang:{code}")
		row.append(btn)
		if len(row) == 2:
			lang_buttons.append(row)
			row = []
	if row:
		lang_buttons.append(row)

	keyboard = [
		[InlineKeyboardButton(text="🤖 Модель Whisper", callback_data="noop")],
		model_buttons,
		[InlineKeyboardButton(text="🌐 Язык перевода", callback_data="noop")],
		*lang_buttons,
	]
	return InlineKeyboardMarkup(inline_keyboard=keyboard)


def _settings_text(user_id: int) -> str:
	s = user_settings.get(user_id)
	lang_label = user_settings.LANGUAGE_LABELS.get(s.language, s.language)
	return (f"*⚙️ Настройки*\n\n"
			f"🤖 Модель: `{s.model}`\n"
			f"🌐 Язык: {lang_label}")


# ── handlers ──────────────────────────────────────────────────────────────────


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
	await message.answer("👋 Привет! Отправь мне голосовое сообщение — я транскрибирую его "
							"с помощью Whisper и переведу на нужный язык.\n\n"
							"Используй /settings чтобы выбрать модель и язык перевода.")


@dp.message(Command("settings"))
async def cmd_settings(message: Message) -> None:
	uid = message.from_user.id
	await message.answer(
		_settings_text(uid),
		reply_markup=_settings_keyboard(uid),
	)


@dp.callback_query(F.data == "noop")
async def cb_noop(callback: CallbackQuery) -> None:
	"""Section header buttons do nothing."""
	await callback.answer()


@dp.callback_query(F.data.startswith("model:"))
async def cb_model(callback: CallbackQuery) -> None:
	uid = callback.from_user.id
	model = callback.data.split(":", 1)[1]
	if model not in user_settings.WHISPER_MODELS:
		await callback.answer("Неизвестная модель.")
		return
	user_settings.set_model(uid, model)
	await callback.message.edit_text(
		_settings_text(uid),
		reply_markup=_settings_keyboard(uid),
	)
	await callback.answer(f"Модель: {model}")


@dp.callback_query(F.data.startswith("lang:"))
async def cb_lang(callback: CallbackQuery) -> None:
	uid = callback.from_user.id
	lang = callback.data.split(":", 1)[1]
	valid_codes = {code for code, _ in user_settings.LANGUAGES}
	if lang not in valid_codes:
		await callback.answer("Неизвестный язык.")
		return
	user_settings.set_language(uid, lang)
	await callback.message.edit_text(
		_settings_text(uid),
		reply_markup=_settings_keyboard(uid),
	)
	label = user_settings.LANGUAGE_LABELS.get(lang, lang)
	await callback.answer(f"Язык: {label}")


@dp.message(F.voice)
async def handle_voice(message: Message) -> None:
	"""Full pipeline: download → transcribe → translate → reply."""
	uid = message.from_user.id
	s = user_settings.get(uid)
	voice_duration: int = message.voice.duration or 1

	# 1. Instant acknowledgement
	status = await message.reply("⏳ Сообщение получено, обрабатываю…")

	tmp_path: str | None = None
	try:
		# 2. Download voice file
		voice_file = await bot.get_file(message.voice.file_id)
		with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
			tmp_path = tmp.name
		await bot.download_file(voice_file.file_path, destination=tmp_path)

		# Cache audio bytes for /benchmark
		with open(tmp_path, "rb") as f:
			voice_bytes = f.read()
		user_settings.set_last_voice(uid, voice_bytes, voice_duration)

		# 3. Transcribe — measure time
		await status.edit_text(f"🎙️ Транскрибирую аудио (модель: `{s.model}`)…")
		t0 = time.monotonic()
		transcribed: str = await asyncio.to_thread(transcriber.transcribe, tmp_path, s.model)
		transcribe_elapsed = time.monotonic() - t0

		if not transcribed:
			await status.edit_text("❌ Не удалось распознать речь. Попробуй ещё раз.")
			return

		# 4. Translate
		lang_label = user_settings.LANGUAGE_LABELS.get(s.language, s.language)
		await status.edit_text(f"🌐 Перевожу на {lang_label}…")
		translated: str = await asyncio.to_thread(translator.translate, transcribed, s.language)

		# 5. Final reply with transcription time
		rtf = transcribe_elapsed / voice_duration
		# timing_line = (
		#     f"⏱ Транскрипция: {transcribe_elapsed:.1f} с "
		#     f"({rtf:.2f}× RT, модель `{s.model}`)"
		# )
		await status.edit_text(f"✅ *Перевод* ({lang_label}):\n\n{translated}\n\n"
								f"_Оригинал:_\n\n {transcribed}\n\n{timing_line}")

	except Exception as exc:
		logger.exception("Error processing voice: %s", exc)
		await status.edit_text(f"❌ Ошибка: `{type(exc).__name__}: {exc}`")
	finally:
		if tmp_path:
			try:
				os.unlink(tmp_path)
			except OSError:
				pass


@dp.message(Command("benchmark"))
async def cmd_benchmark(message: Message) -> None:
	"""Run all Whisper models on the last received voice, report timing."""
	uid = message.from_user.id
	s = user_settings.get(uid)

	if s.last_voice_bytes is None:
		await message.answer("❌ Нет сохранённого аудио.\n"
								"Сначала отправь голосовое сообщение, потом /benchmark.")
		return

	duration = s.last_voice_duration
	models = user_settings.WHISPER_MODELS
	results: dict[str, tuple[float, float]] = {}  # model → (elapsed_s, rtf)

	def build_status(running: str | None) -> str:
		lines = [f"🧪 *Бенчмарк Whisper* | аудио: {duration} с\n"]
		for m in models:
			if m in results:
				elapsed, rtf = results[m]
				lines.append(f"✅ `{m}`: {elapsed:.1f} с ({rtf:.2f}× RT)")
			elif m == running:
				lines.append(f"⏳ `{m}`…")
			else:
				lines.append(f"⬜ `{m}`")
		return "\n".join(lines)

	status = await message.answer(f"🧪 *Бенчмарк Whisper* | аудио: {duration} с\n\n⏳ Запускаю…")

	tmp_path: str | None = None
	try:
		with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
			tmp.write(s.last_voice_bytes)
			tmp_path = tmp.name

		async def safe_edit(text: str) -> None:
			try:
				await status.edit_text(text)
			except TelegramBadRequest as e:
				if "message is not modified" not in str(e):
					raise

		for i, model in enumerate(models):
			await safe_edit(build_status(model))
			t0 = time.monotonic()
			await asyncio.to_thread(transcriber.transcribe, tmp_path, model)
			elapsed = time.monotonic() - t0
			rtf = elapsed / max(duration, 1)
			results[model] = (elapsed, rtf)
			# Update after each model finishes
			next_model = models[i + 1] if i + 1 < len(models) else None
			await safe_edit(build_status(next_model))

	except Exception as exc:
		logger.exception("Benchmark error: %s", exc)
		await status.edit_text(f"❌ Ошибка: `{type(exc).__name__}: {exc}`")
	finally:
		if tmp_path:
			try:
				os.unlink(tmp_path)
			except OSError:
				pass


# ── entry point ───────────────────────────────────────────────────────────────


async def main() -> None:
	logger.info("Starting bot (default model: %s). Listening…", config.WHISPER_MODEL)
	await dp.start_polling(bot, allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
	asyncio.run(main())
