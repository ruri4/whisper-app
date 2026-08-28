import asyncio
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import discord
from discord import app_commands

from whisper_app.transcribe import transcribe_source

_thread_pool = ThreadPoolExecutor(max_workers=2)


async def transcribe_source_async(source: str, model_size: str = "large-v3-turbo") -> tuple[str, str, float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_thread_pool, transcribe_source, source, model_size)


AUDIO_EXTENSIONS = (
    ".ogg",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".mp4",
    ".mov",
    ".webm",
    ".opus",
    ".aac",
)


def _find_audio_in_attachments_or_text(attachments, text: str) -> Optional[str]:
    for att in attachments:
        if att.content_type and (
            att.content_type.startswith("audio/")
            or att.content_type.startswith("video/")
        ) or att.filename.lower().endswith(AUDIO_EXTENSIONS):
            return att.url

    if text:
        words = text.split()
        for w in words:
            if w.startswith(("http://", "https://")):
                clean_w = w.split("?")[0].lower()
                if clean_w.endswith(AUDIO_EXTENSIONS):
                    return w

    return None


def _extract_audio_url_from_message(message: discord.Message) -> Optional[str]:
    # 1. Direct message attachments and content
    direct = _find_audio_in_attachments_or_text(message.attachments, message.content)
    if direct:
        return direct

    # 2. Forwarded message snapshots (Discord forwarded messages)
    if hasattr(message, "message_snapshots") and message.message_snapshots:
        for snap in message.message_snapshots:
            snap_audio = _find_audio_in_attachments_or_text(snap.attachments, snap.content)
            if snap_audio:
                return snap_audio

    return None


class WhisperBot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()
        print("synced slash commands globally")


bot = WhisperBot()


@bot.tree.command(
    name="transcribe",
    description="Transcribe audio/video from a URL or attachment",
)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(
    url="URL to the audio/video file",
    attachment="Audio/video file to transcribe",
    model="Whisper model size",
)
@app_commands.choices(
    model=[
        app_commands.Choice(name="large-v3-turbo (default, fast & accurate)", value="large-v3-turbo"),
        app_commands.Choice(name="tiny (fastest)", value="tiny"),
        app_commands.Choice(name="base", value="base"),
        app_commands.Choice(name="small", value="small"),
        app_commands.Choice(name="medium", value="medium"),
        app_commands.Choice(name="large-v3", value="large-v3"),
    ]
)
async def transcribe_slash(
    interaction: discord.Interaction,
    url: Optional[str] = None,
    attachment: Optional[discord.Attachment] = None,
    model: app_commands.Choice[str] = None,
):
    target_url = None
    if attachment is not None:
        target_url = attachment.url
    elif url is not None:
        target_url = url

    if not target_url:
        await interaction.response.send_message(
            "please provide either an `attachment` or a `url`.", ephemeral=True
        )
        return

    model_name = model.value if model else "large-v3-turbo"
    await interaction.response.defer(ephemeral=False, thinking=True)

    try:
        text, lang, prob = await transcribe_source_async(target_url, model_size=model_name)
        if not text:
            await interaction.followup.send(
                f"*(no speech detected / detected language: {lang} with {prob:.2f} prob)*"
            )
            return

        if len(text) > 1900:
            text = text[:1900] + "... [truncated]"

        await interaction.followup.send(text)
    except Exception as e:
        await interaction.followup.send(f"transcription failed: `{e}`", ephemeral=True)


@bot.tree.context_menu(
    name="Transcribe Message",
)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def transcribe_context_menu(interaction: discord.Interaction, message: discord.Message):
    target_url = _extract_audio_url_from_message(message)
    if not target_url:
        await interaction.response.send_message(
            "no audio attachment or direct audio link found in this message.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(ephemeral=False, thinking=True)

    try:
        text, lang, prob = await transcribe_source_async(target_url, model_size="large-v3-turbo")
        if not text:
            await interaction.followup.send(
                f"*(no speech detected / detected language: {lang} with {prob:.2f} prob)*"
            )
            return

        if len(text) > 1900:
            text = text[:1900] + "... [truncated]"

        await interaction.followup.send(text)
    except Exception as e:
        await interaction.followup.send(f"transcription failed: `{e}`", ephemeral=True)


def run_bot(token: str):
    bot.run(token)
