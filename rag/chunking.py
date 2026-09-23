"""Conservative, source-preserving chunks; no Rust AST dependency required.

The code chunker recognizes function regions, not a complete Verus grammar.
Unrecognized regions remain searchable instead of being silently discarded.
"""
from __future__ import annotations

import re


def mask_non_code(text: str) -> str:
    """Mask comments and Rust strings while preserving offsets and newlines."""
    raw_pattern = re.compile(r'(?:br|r)(#{0,255})"')
    char_pattern = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\n])'")
    chars = list(text)
    i = 0
    while i < len(text):
        start = i
        if text.startswith('//', i):
            end = text.find('\n', i)
            i = len(text) if end < 0 else end
        elif text.startswith('/*', i):
            i += 2
            depth = 1
            while i < len(text) and depth:
                if text.startswith('/*', i):
                    depth += 1
                    i += 2
                elif text.startswith('*/', i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
        else:
            raw = raw_pattern.match(text, i) if text[i] in 'br' else None
            char = char_pattern.match(text, i) if text[i] == "'" else None
            if raw:
                terminator = '"' + raw.group(1)
                end = text.find(terminator, raw.end())
                i = len(text) if end < 0 else end + len(terminator)
            elif text[i] == '"':
                i += 1
                while i < len(text):
                    if text[i] == '\\':
                        i += 2
                    elif text[i] == '"':
                        i += 1
                        break
                    else:
                        i += 1
            elif char:
                i = char.end()
            else:
                i += 1
                continue
        for j in range(start, min(i, len(chars))):
            if chars[j] != '\n':
                chars[j] = ' '
    return ''.join(chars)


def code_chunks(text: str, fallback_lines: int = 160) -> list[tuple[int, int, str]]:
    """Keep function regions intact, even when longer than a line window.

    Boundaries are function declaration lines found outside comments/strings.
    This intentionally includes trailing module braces and adjacent declarations;
    it is a conservative lexical region, not an AST-certified function span.
    """
    lines = text.splitlines(keepends=True)
    masked = mask_non_code(text).splitlines()
    starts = [i for i, line in enumerate(masked) if re.search(r'\bfn\s+[A-Za-z_]\w*', line)]
    if not starts:
        return [(i + 1, min(i + fallback_lines, len(lines)), ''.join(lines[i:i + fallback_lines]))
                for i in range(0, len(lines), fallback_lines) if ''.join(lines[i:i + fallback_lines]).strip()]
    # Include adjacent doc comments and attributes with the declaration.
    for n, start in enumerate(starts):
        lower = starts[n - 1] + 1 if n else 0
        while start > lower and (lines[start - 1].lstrip().startswith(('///', '//!', '#['))):
            start -= 1
        starts[n] = start
    boundaries = sorted(set([0, *starts, len(lines)]))
    return [(a + 1, b, ''.join(lines[a:b])) for a, b in zip(boundaries, boundaries[1:])
            if ''.join(lines[a:b]).strip()]


def document_chunks(text: str, target_chars: int = 1800) -> list[tuple[int, int, str]]:
    """Pack paragraphs/sections without splitting fenced code or losing lines."""
    lines = text.splitlines(keepends=True)
    chunks = []
    start = 0
    size = 0
    fence = None
    for i, line in enumerate(lines):
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        heading = fence is None and re.match(r'^#{1,6}\s', line)
        if i > start and fence is None and (heading or (size >= target_chars and not line.strip())):
            block = ''.join(lines[start:i])
            if block.strip():
                chunks.append((start + 1, i, block))
            start, size = i, 0
        if marker:
            mark = marker.group(1)
            if fence is None:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence) and not line[marker.end():].strip():
                fence = None
        size += len(line)
    block = ''.join(lines[start:])
    if block.strip():
        chunks.append((start + 1, len(lines), block))
    return chunks
