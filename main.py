import argparse
import ctypes
import os
import site
import sys
from pathlib import Path


def _preload_cuda_libraries():
    for sp in site.getsitepackages():
        nvidia_dir = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for sub in os.listdir(nvidia_dir):
            lib_dir = os.path.join(nvidia_dir, sub, "lib")
            if not os.path.isdir(lib_dir):
                continue
            for f in os.listdir(lib_dir):
                if f.endswith(".so") or ".so." in f:
                    try:
                        ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass


_preload_cuda_libraries()

from faster_whisper import WhisperModel
import ctranslate2


import argparse
import ctypes
import io
import os
import site
import sys
import tempfile
import urllib.request
from pathlib import Path


def _preload_cuda_libraries():
    for sp in site.getsitepackages():
        nvidia_dir = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for sub in os.listdir(nvidia_dir):
            lib_dir = os.path.join(nvidia_dir, sub, "lib")
            if not os.path.isdir(lib_dir):
                continue
            for f in os.listdir(lib_dir):
                if f.endswith(".so") or ".so." in f:
                    try:
                        ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass


_preload_cuda_libraries()

from faster_whisper import WhisperModel
import ctranslate2


def transcribe_source(source: str, model_size: str = "base", device: str = "auto", compute_type: str = "auto"):
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    audio_input = source
    temp_file = None

    if source.startswith(("http://", "https://")):
        print(f"fetching audio from url...", file=sys.stderr)
        req = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
        temp_file.write(data)
        temp_file.flush()
        temp_file.close()
        audio_input = temp_file.name

    try:
        print(f"loading model '{model_size}' on {device} ({compute_type})...", file=sys.stderr)
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        segments, info = model.transcribe(audio_input, beam_size=5)
        print(f"detected language '{info.language}' with probability {info.language_probability:.2f}", file=sys.stderr)

        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    finally:
        if temp_file is not None and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio from file or URL using faster-whisper")
    parser.add_argument("source", nargs="?", help="Path to audio file or HTTP(S) URL")
    parser.add_argument("--model", default="base", help="Model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--compute-type", default="auto", help="Compute type (e.g. float16, int8, auto)")

    args = parser.parse_args()

    if not args.source:
        parser.print_help()
        sys.exit(0)

    is_url = args.source.startswith(("http://", "https://"))
    if not is_url and not Path(args.source).is_file():
        print(f"error: file not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    transcribe_source(args.source, model_size=args.model, device=args.device, compute_type=args.compute_type)


if __name__ == "__main__":
    main()
