"""Shared ingest source validation (path-traversal safe)."""
from __future__ import annotations

from pathlib import Path

ALLOWED_SUFFIXES = {".pdf", ".md", ".markdown", ".png", ".jpg", ".jpeg", ".webp"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def validate_source(source: str, settings) -> str:
    if source.startswith(("http://", "https://")):
        return source
    data_dir = Path(settings.DATA_DIR).resolve()
    file_path = Path(source).resolve()
    if not file_path.is_relative_to(data_dir):
        raise ValueError(f"File path must be within DATA_DIR ({data_dir})")
    if not file_path.exists():
        raise ValueError(f"File not found: {source}")
    if file_path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {size_mb:.1f}MB (max {settings.MAX_FILE_SIZE_MB}MB)")
    return source
