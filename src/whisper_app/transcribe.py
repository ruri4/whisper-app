import ctypes
import os
import site
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


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

import ctranslate2
from faster_whisper import WhisperModel

_models: dict[str, WhisperModel] = {}
_thread_pool = ThreadPoolExecutor(max_workers=2)


def get_model(model_size: str = "large-v3-turbo") -> WhisperModel:
    if model_size not in _models:
        cuda_avail = ctranslate2.get_cuda_device_count() > 0
        device = "cuda" if cuda_avail else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"loading model '{model_size}' on {device} ({compute_type})...", file=sys.stderr)
        _models[model_size] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _models[model_size]


def transcribe_source(source: str, model_size: str = "large-v3-turbo") -> tuple[str, str, float]:
    audio_input = source
    temp_file = None

    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()

        temp_file = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
        temp_file.write(data)
        temp_file.flush()
        temp_file.close()
        audio_input = temp_file.name

    try:
        model = get_model(model_size)
        segments, info = model.transcribe(
            audio_input,
            beam_size=1,
            vad_filter=True,
        )

        texts = [s.text.strip() for s in segments if s.text.strip()]
        full_text = " ".join(texts).strip()
        return full_text, info.language, info.language_probability
    finally:
        if temp_file is not None and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
