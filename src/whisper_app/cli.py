import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from whisper_app.transcribe import transcribe_source

load_dotenv()


def run_cli(args):
    is_url = args.source.startswith(("http://", "https://"))
    if not is_url and not Path(args.source).is_file():
        print(f"error: file not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    text, _lang, _prob = transcribe_source(args.source, model_size=args.model)
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio or run Discord bot")
    parser.add_argument("source", nargs="?", help="Path to audio file or HTTP(S) URL (CLI mode)")
    parser.add_argument("--bot", action="store_true", help="Run Discord user app / bot")
    parser.add_argument("--model", default="large-v3-turbo", help="Model size (tiny, base, small, medium, large-v3, large-v3-turbo)")

    args = parser.parse_args()

    if args.bot or (not args.source and os.environ.get("DISCORD_TOKEN")):
        token = os.environ.get("DISCORD_TOKEN")
        if not token:
            print("error: DISCORD_TOKEN environment variable is required to run the bot.", file=sys.stderr)
            sys.exit(1)
        print("starting discord user app bot...", file=sys.stderr)
        from whisper_app.bot import run_bot
        run_bot(token)
    elif args.source:
        run_cli(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
