"""
Telegram bot — aiogram 3.x rewrite.

Features
--------
* Accepts voice messages in private chats and groups.
* Live status editing: received → transcribing → translating → result.
* /settings command: inline keyboard to switch Whisper model and target language.
* /benchmark command: runs all Whisper models on last received voice, reports timing.
* /admin command: restricted to ADMIN_CHAT_ID — shows loaded models & active tasks,
  with buttons to unload a model or stop a task.
* Per-user settings stored in memory (settings.py).
* Fully async: blocking Whisper inference and HTTP calls run in thread pool.
"""

import asyncio
import logging
import os
import tempfile
import time
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import BaseFilter, Command
from aiogram.types import (
	CallbackQuery,
	InlineKeyboardButton,
	InlineKeyboardMarkup,
	Message,
	TelegramObject,
	Update,
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

# ── active task registry ──────────────────────────────────────────────────────
# key: "uid:<user_id>:msg:<message_id>" → asyncio.Task
_active_tasks: dict[str, asyncio.Task] = {}


def _task_key(user_id: int, message_id: int) -> str:
	return f"uid:{user_id}:msg:{message_id}"


def _register_task(key: str, task: asyncio.Task) -> None:
	_active_tasks[key] = task
	task.add_done_callback(lambda t: _active_tasks.pop(key, None))


# ── pending voice registry ────────────────────────────────────────────────────
# Store voice message + status message while waiting for language selection
# key: "uid:<user_id>:msg:<message_id>" → (message, status_message)
_pending_voices: dict[str, tuple[Message, Message]] = {}


def _voice_key(user_id: int, message_id: int) -> str:
	return f"uid:{user_id}:msg:{message_id}"


def _register_pending_voice(key: str, message: Message, status: Message) -> None:
	_pending_voices[key] = (message, status)


def _retrieve_pending_voice(key: str) -> tuple[Message, Message] | None:
	return _pending_voices.pop(key, None)


# ── access control middleware ──────────────────────────────────────────────────

class AllowlistMiddleware(BaseMiddleware):
	"""Allow everyone to use /start; restrict everything else to ALLOWED_CHAT_IDS."""

	async def __call__(
		self,
		handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
		event: TelegramObject,
		data: dict[str, Any],
	) -> Any:
		update: Update = data.get("update") or event  # type: ignore[assignment]

		# Determine the chat id from whatever update type we received
		chat_id: int | None = None
		if isinstance(event, Message):
			chat_id = event.chat.id
		elif isinstance(event, CallbackQuery) and event.message:
			chat_id = event.message.chat.id

		# Detect /start commands — they are allowed for everyone
		is_start = (
			isinstance(event, Message)
			and event.text
			and event.text.strip().startswith("/start")
		)

		if is_start:
			return await handler(event, data)

		# Admin commands / callbacks are gated by IsAdmin filter, not this middleware
		is_admin_event = False
		if isinstance(event, Message) and event.text:
			is_admin_event = event.text.strip().startswith("/admin")
		elif isinstance(event, CallbackQuery) and event.data:
			is_admin_event = event.data.startswith("admin:")

		if is_admin_event:
			return await handler(event, data)

		if chat_id is not None and chat_id not in config.ALLOWED_CHAT_IDS:
			# Silently ignore — do not respond
			return

		return await handler(event, data)


dp.update.middleware(AllowlistMiddleware())


# ── admin filter ──────────────────────────────────────────────────────────────

class IsAdmin(BaseFilter):
	"""Pass only when the event originates from ADMIN_CHAT_ID."""

	async def __call__(self, event: Message | CallbackQuery) -> bool:
		if config.ADMIN_CHAT_ID is None:
			return False
		if isinstance(event, Message):
			return event.chat.id == config.ADMIN_CHAT_ID
		if isinstance(event, CallbackQuery) and event.message:
			return event.message.chat.id == config.ADMIN_CHAT_ID
		return False


# ── admin panel helpers ───────────────────────────────────────────────────────

def _admin_text() -> str:
	models = transcriber.list_loaded_models()
	tasks = list(_active_tasks.keys())

	lines = ["🛠 *Admin Panel*\n"]

	if models:
		lines.append(f"🤖 *Loaded models* ({len(models)}): " + ", ".join(f"`{m}`" for m in models))
	else:
		lines.append("🤖 *Loaded models*: none")

	if tasks:
		lines.append(f"\n📋 *Active tasks* ({len(tasks)}):")
		for key in tasks:
			lines.append(f"  • `{key}`")
	else:
		lines.append("\n📋 *Active tasks*: none")

	return "\n".join(lines)


def _admin_keyboard() -> InlineKeyboardMarkup:
	rows: list[list[InlineKeyboardButton]] = []

	# Unload buttons (one per loaded model)
	models = transcriber.list_loaded_models()
	if models:
		rows.append([
			InlineKeyboardButton(
				text=f"🗑 Unload {m}",
				callback_data=f"admin:unload:{m}",
			)
			for m in models
		])

	# Stop buttons (one per active task)
	for key in list(_active_tasks.keys()):
		# Shorten long keys for button labels
		label = key if len(key) <= 32 else key[:29] + "…"
		rows.append([
			InlineKeyboardButton(
				text=f"⛔ Stop {label}",
				callback_data=f"admin:stop:{key}",
			)
		])

	# Refresh button always present
	rows.append([InlineKeyboardButton(text="🔄 Refresh", callback_data="admin:refresh")])

	return InlineKeyboardMarkup(inline_keyboard=rows)


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
	if message.chat.id not in config.ALLOWED_CHAT_IDS:
		await message.answer("👋 Привет! Этот бот доступен только для выбранных пользователей.")
		return
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


# ── admin command & callbacks ─────────────────────────────────────────────────

@dp.message(IsAdmin(), Command("admin"))
async def cmd_admin(message: Message) -> None:
	await message.answer(_admin_text(), reply_markup=_admin_keyboard())


@dp.callback_query(IsAdmin(), F.data == "admin:refresh")
async def cb_admin_refresh(callback: CallbackQuery) -> None:
	try:
		await callback.message.edit_text(_admin_text(), reply_markup=_admin_keyboard())
	except TelegramBadRequest as e:
		if "message is not modified" not in str(e):
			raise
	await callback.answer("Refreshed")


@dp.callback_query(IsAdmin(), F.data.startswith("admin:unload:"))
async def cb_admin_unload(callback: CallbackQuery) -> None:
	model = callback.data.split(":", 2)[2]
	removed = await asyncio.to_thread(transcriber.unload_model, model)
	msg = f"Model `{model}` unloaded." if removed else f"Model `{model}` was not loaded."
	try:
		await callback.message.edit_text(_admin_text(), reply_markup=_admin_keyboard())
	except TelegramBadRequest as e:
		if "message is not modified" not in str(e):
			raise
	await callback.answer(msg)


@dp.callback_query(IsAdmin(), F.data.startswith("admin:stop:"))
async def cb_admin_stop(callback: CallbackQuery) -> None:
	key = callback.data.split(":", 2)[2]
	task = _active_tasks.get(key)
	if task and not task.done():
		task.cancel()
		msg = f"Task stopped."
	else:
		msg = "Task not found or already finished."
	# Small yield so done_callback has a chance to remove from registry
	await asyncio.sleep(0)
	try:
		await callback.message.edit_text(_admin_text(), reply_markup=_admin_keyboard())
	except TelegramBadRequest as e:
		if "message is not modified" not in str(e):
			raise
	await callback.answer(msg)


# ── non-admin admin callbacks — silently ignore ───────────────────────────────

@dp.callback_query(F.data.startswith("admin:"))
async def cb_admin_unauthorized(callback: CallbackQuery) -> None:
	await callback.answer("⛔ Not authorized.", show_alert=True)


# ── settings callbacks ────────────────────────────────────────────────────────

@dp.callback_query(F.data == "noop")
async def cb_noop(callback: CallbackQuery) -> None:
	"""Section header buttons do nothing."""
	await callback.answer()


# ── voice language selection callbacks ────────────────────────────────────────

def _voice_lang_keyboard() -> InlineKeyboardMarkup:
	"""Build inline keyboard for initial language selection."""
	return InlineKeyboardMarkup(inline_keyboard=[
		[
			InlineKeyboardButton(text="Сингальский", callback_data="voice_lang:si"),
			InlineKeyboardButton(text="Автоопредление", callback_data="voice_lang:auto"),
		]
	])


@dp.callback_query(F.data.startswith("voice_lang:"))
async def cb_voice_lang(callback: CallbackQuery) -> None:
	"""Handle initial language selection for voice processing."""
	uid = callback.from_user.id
	lang_choice = callback.data.split(":", 1)[1]
	
	# Extract message_id from callback to match pending voice
	if not callback.message or not callback.message.reply_to_message:
		await callback.answer("❌ Не удалось найти голосовое сообщение.")
		return
	
	voice_msg_id = callback.message.reply_to_message.message_id
	key = _voice_key(uid, voice_msg_id)
	
	pending_voice_data = _retrieve_pending_voice(key)
	if pending_voice_data is None:
		await callback.answer("❌ Голосовое сообщение больше не доступно.")
		return
	
	message, old_status = pending_voice_data
	
	try:
		# Delete the language selection message
		try:
			await callback.message.delete()
		except Exception:
			pass
		
		# Create a fresh status message for processing
		new_status = await message.reply("⏳ Сообщение получено, обрабатываю…")
		
		# Determine transcription mode based on user selection
		mode = "si" if lang_choice == "si" else "default"
		
		# Start processing with the selected mode
		process_task = asyncio.create_task(
			_process_voice(message, new_status, mode),
			name=key,
		)
		_register_task(key, process_task)
		
		await callback.answer("✅")
	
	except Exception as e:
		logger.exception("Error in voice language selection: %s", e)
		await callback.answer("❌ Ошибка при обработке.")


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


# ── voice handler ─────────────────────────────────────────────────────────────

async def _process_voice(
	message: Message,
	status: Message,
	transcription_mode: str = "default",
) -> None:
	"""Full pipeline: download → transcribe → translate → reply.
	
	Args:
		message: The voice message.
		status: The status message to edit.
		transcription_mode: "si" for Sinhala model, "default" for standard Whisper.
	"""
	uid = message.from_user.id
	s = user_settings.get(uid)
	voice_duration: int = message.voice.duration or 1
	
	# Determine transcription language based on mode
	if transcription_mode == "si":
		language_for_transcription = "si"
	else:
		language_for_transcription = s.language

	tmp_path: str | None = None
	try:
		# 1. Download voice file
		voice_file = await bot.get_file(message.voice.file_id)
		with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
			tmp_path = tmp.name
		await bot.download_file(voice_file.file_path, destination=tmp_path)

		# Cache audio bytes for /benchmark
		with open(tmp_path, "rb") as f:
			voice_bytes = f.read()
		user_settings.set_last_voice(uid, voice_bytes, voice_duration)

		# 2. Transcribe — measure time
		await status.edit_text(f"🎙️ Транскрибирую аудио (модель: `{s.model}`)…")
		t0 = time.monotonic()
		transcribed: str = await asyncio.to_thread(
			transcriber.transcribe,
			tmp_path,
			s.model,
			language=language_for_transcription,
		)
		transcribe_elapsed = time.monotonic() - t0

		if not transcribed:
			await status.edit_text("❌ Не удалось распознать речь. Попробуй ещё раз.")
			return

		# 3. Translate
		# Determine which language to show in UI
		display_lang = language_for_transcription if language_for_transcription is not None else s.language
		lang_label = user_settings.LANGUAGE_LABELS.get(display_lang, display_lang)
		await status.edit_text(f"🌐 Перевожу на {lang_label}…")
		translated: str = await asyncio.to_thread(translator.translate, transcribed, display_lang)

		# 4. Final reply
		rtf = transcribe_elapsed / voice_duration
		await status.edit_text(f"✅ *Перевод* ({lang_label}):\n\n{translated}\n\n"
							   f"_Оригинал:_\n\n {transcribed}")

	except asyncio.CancelledError:
		logger.info("Voice task cancelled for user %s", uid)
		try:
			await status.edit_text("⛔ Обработка остановлена администратором.")
		except Exception:
			pass
		raise

	except Exception as exc:
		logger.exception("Error processing voice: %s", exc)
		try:
			await status.edit_text(f"❌ Ошибка: `{type(exc).__name__}: {exc}`")
		except Exception:
			pass

	finally:
		if tmp_path:
			try:
				os.unlink(tmp_path)
			except OSError:
				pass


@dp.message(F.voice)
async def handle_voice(message: Message) -> None:
	uid = message.from_user.id
	# Send language selection message
	status = await message.reply(
		"🌐 Выбери язык:",
		reply_markup=_voice_lang_keyboard(),
	)
	
	key = _voice_key(uid, message.message_id)
	_register_pending_voice(key, message, status)


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
	# Use user's default language for benchmark
	bench_language = s.language

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
			await asyncio.to_thread(transcriber.transcribe, tmp_path, model, language=bench_language)
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
