"""One context renderer shared by library and CLI; summaries are UI-only."""
from __future__ import annotations


def render_prompt_context(result: dict, max_items: int = 12, max_chars: int = 24000) -> str:
    """Pack complete evidence blocks, skipping blocks that do not fit.

    Character budget is intentionally explicit; it is not a model token budget.
    Callers can read full result text even when a block exceeds this budget.
    """
    if max_items < 0 or max_chars < 0:
        raise ValueError('Context limits must be nonnegative')
    blocks = []
    used = 0
    for item in result.get('results', [])[:max_items]:
        location = f"{item['path']}:{item.get('line_start', 1)}-{item.get('line_end', 1)}"
        if item.get('page'):
            location = f"{item['path']}#page-{item['page']}"
        text = item.get('text', item.get('snippet', ''))
        for ref in item.get("includes", []):
            if ref.get("status") == "expanded":
                location += f"; includes {ref['path']}:{ref['line_start']}-{ref['line_end']}"
        block = f"[Evidence {item['id']}] {location}\n{text}\n"
        if used + len(block) + (1 if blocks else 0) > max_chars:
            continue
        blocks.append(block)
        used += len(block) + (1 if len(blocks) > 1 else 0)
    return '\n'.join(blocks)
