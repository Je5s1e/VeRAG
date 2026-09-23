"""Resolve local mdBook code includes without changing the citing document span."""
from __future__ import annotations

import re
from pathlib import Path

INCLUDE = re.compile(r'\{\{#include\s+([^}]+)\}\}')


def expand_includes(text: str, document: Path, root: Path) -> tuple[str, list[dict]]:
    references = []

    def replace(match):
        reference = match.group(1).strip()
        parts = reference.split(':', 1)
        path = (document.parent / parts[0]).resolve()
        record = {'reference': reference, 'status': 'missing'}
        references.append(record)
        if not path.is_relative_to(root.resolve()):
            record['status'] = 'outside_source_root'
            return '[Example unavailable: include is outside the source snapshot]'
        if not path.is_file():
            return '[Example unavailable: source file missing from snapshot]'
        lines = path.read_text(encoding='utf-8').splitlines()
        start, end = 0, len(lines)
        if len(parts) == 2:
            anchor = parts[1]
            begins = [i for i, line in enumerate(lines) if re.search(r'ANCHOR:\s*' + re.escape(anchor) + r'\s*$', line)]
            ends = [i for i, line in enumerate(lines) if re.search(r'ANCHOR_END:\s*' + re.escape(anchor) + r'\s*$', line)]
            if not begins or not ends or ends[0] <= begins[0]:
                record['status'] = 'unsupported_or_missing_anchor'
                return '[Example unavailable: include anchor could not be resolved]'
            start, end = begins[0] + 1, ends[0]
        record.update(status='expanded', path=str(path.relative_to(root.resolve())), line_start=start + 1, line_end=end)
        return '\n'.join(lines[start:end])

    return INCLUDE.sub(replace, text), references
