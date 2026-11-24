"""Хранение сырых ответов по SHA-256 и запись метаданных."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def snapshot(root: Path, content: bytes, suffix: str, metadata: dict) -> Path:
    digest = hashlib.sha256(content).hexdigest()
    path = root / f"{digest}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
        write_json(
            path.with_suffix(path.suffix + ".json"),
            {
                **metadata,
                "sha256": digest,
                "downloaded_at_utc": datetime.now(UTC).isoformat(),
            },
        )
    return path


def frame_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
