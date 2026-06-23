#!/usr/bin/env python3
"""
Tag one or more notebook cells as 'freeze' by cell ID.

Frozen cells are skipped by execute_notebook.py, preserving their saved output
across full notebook re-executions. Use this immediately after you get an LLM
output you want to keep.

Usage:
    python scripts/freeze_cell.py <notebook.ipynb> <cell_id> [<cell_id> ...]

Example:
    python scripts/freeze_cell.py chapters/intro.ipynb ab45c824 chess-diagram
"""

import json
import sys
from pathlib import Path


def freeze_cells(notebook_path: str, cell_ids: list[str]) -> None:
    path = Path(notebook_path)
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)

    remaining = set(cell_ids)
    for cell in nb["cells"]:
        cid = cell.get("id")
        if cid in remaining:
            tags = cell.setdefault("metadata", {}).setdefault("tags", [])
            if "freeze" not in tags:
                tags.append("freeze")
                print(f"  [frozen] {cid}")
            else:
                print(f"  [already frozen] {cid}")
            remaining.discard(cid)

    if remaining:
        print(f"  [not found] {', '.join(remaining)}", file=sys.stderr)
        sys.exit(1)

    # Preserve the file's on-disk JSON style (unicode escaping + trailing
    # newline) so cells we didn't touch stay byte-for-byte identical.
    new = json.dumps(nb, indent=1, ensure_ascii=raw.isascii())
    if raw.endswith("\n"):
        new += "\n"
    if new != raw:
        path.write_text(new, encoding="utf-8")
    print(f"Saved: {path.name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    freeze_cells(sys.argv[1], sys.argv[2:])
