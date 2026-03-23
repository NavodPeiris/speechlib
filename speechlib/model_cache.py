"""
Model cache management for speechlib.

Centralizes all model caching to avoid duplicates across libraries:
- pyannote.audio
- faster-whisper
- transformers
- torch hub
"""

import os
from pathlib import Path


def get_cache_dir() -> Path:
    """Get the centralized model cache directory."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(xdg_cache) / "huggingface"


def setup_model_cache() -> None:
    """
    Configure environment for shared model cache.

    Call this at module initialization (e.g., in __init__.py)
    before importing ML libraries.

    Usage:
        from speechlib.model_cache import setup_model_cache
        setup_model_cache()

        # Now import ML libraries
        from pyannote.audio import Pipeline
        from faster_whisper import WhisperModel
    """
    cache_dir = get_cache_dir()

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

    os.environ.setdefault("PYTORCH_PRETRAINED_BERT_CACHE", str(cache_dir / "pytorch"))


def print_cache_info() -> None:
    """Print current model cache status."""
    cache_dir = get_cache_dir()
    hub_dir = cache_dir / "hub"

    print(f"\n=== speechlib Model Cache ===")
    print(f"Cache directory: {cache_dir}")

    if hub_dir.exists():
        total_size = sum(
            sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            for item in hub_dir.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        ) / (1024**3)
        print(f"Total size: {total_size:.2f} GB")
        print("\nCached models:")
        for item in sorted(hub_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (
                    1024**3
                )
                print(f"  {item.name}: {size:.2f} GB")
