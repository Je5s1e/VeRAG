"""Immutable index snapshots, atomically published through meta.json."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

SCHEMA_VERSION = 2


def resolve_snapshot(index_dir: str | Path) -> tuple[Path, dict]:
    root = Path(index_dir).resolve()
    meta_path = root / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    version = meta.get('schema_version', 1)
    if version not in (1, SCHEMA_VERSION):
        raise ValueError(f'Unsupported index schema {version}; rebuild the index.')
    generation = meta.get('generation')
    if generation and (not isinstance(generation, str) or not generation.isalnum()):
        raise ValueError('Invalid index generation')
    snapshot = root / 'generations' / generation if generation else root
    if not (snapshot / 'chunks.jsonl').is_file():
        raise FileNotFoundError(f'No index at {root}. Run: verag build --index-dir {root}')
    return snapshot, meta


def publish_manifest(root: Path, metadata: dict) -> None:
    temp = root / f'.meta-{uuid.uuid4().hex}.tmp'
    try:
        temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, root / 'meta.json')
    finally:
        temp.unlink(missing_ok=True)
