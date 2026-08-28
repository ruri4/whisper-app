# whisper-app

GPU-accelerated local transcription CLI and Discord User App powered by `faster-whisper` and CUDA 12.

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support
- `uv` package manager

## Installation

```bash
uv sync
```

## Configuration

For Discord bot / user app mode, create a `.env` file or export `DISCORD_TOKEN`:

```env
DISCORD_TOKEN=your_bot_token_here
```

## Usage

### CLI

Transcribe local audio files or direct audio/video URLs:

```bash
# using uv run script entrypoint (defaults to large-v3-turbo)
uv run whisper-app <audio-file-or-url>

# with custom model
uv run whisper-app <audio-file-or-url> --model base
```

Available model sizes: `large-v3-turbo` (default), `large-v3`, `medium`, `small`, `base`, `tiny`.

### Discord User App

Start the bot:

```bash
uv run whisper-app --bot
# or
uv run python -m whisper_app.cli --bot
```

**Features:**
- `/transcribe [url] [attachment] [model]`: Slash command configured for user installation (`allowed_installs=[users, guilds]`, `allowed_contexts=[guilds, dms, private_channels]`). Usable anywhere on Discord.
- **Message Context Menu ("Apps -> Transcribe Message")**: Right click or long press any voice note or message containing audio attachments/links to transcribe with `large-v3`.
