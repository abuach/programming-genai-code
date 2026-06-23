#!/usr/bin/env python3
"""
Execute a Jupyter notebook while preserving outputs of cells tagged 'freeze'.

Frozen cells are still skipped entirely — their code does NOT run in the
kernel — so avoid freezing cells whose variables are needed by later cells.
The intended use case is non-deterministic LLM output cells whose printed
result you want to lock in permanently.

Usage:
    python scripts/execute_notebook.py chapters/foo.ipynb
    python scripts/execute_notebook.py chapters/foo.ipynb --trim   # also trim outputs

To freeze a cell, add "freeze" to its tags in the notebook metadata:
    cell["metadata"]["tags"] = ["freeze"]
"""

import os
import sys
from pathlib import Path

import nbclient
import nbformat


class FreezableNotebookClient(nbclient.NotebookClient):
    """NotebookClient that skips cells tagged 'freeze', keeping saved outputs."""

    async def async_execute_cell(self, cell, cell_index, **kwargs):
        tags = cell.get("metadata", {}).get("tags", [])
        if "freeze" in tags:
            print(f"  [frozen] cell {cell_index} — keeping saved output")
            return cell
        return await super().async_execute_cell(cell, cell_index, **kwargs)


def execute(notebook_path: str, trim: bool = False) -> None:
    path = Path(notebook_path).resolve()
    nb = nbformat.read(path, as_version=4)

    # Kernel inherits this process's env; make genai importable from chapters/
    os.environ.setdefault("PYTHONPATH", str(path.parent))

    client = FreezableNotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    nbformat.write(nb, path)
    print(f"Executed: {path.name}")

    # Reformat any labelled transcripts (genai.show_turn et al.) to the book's
    # auto-gutter, left-aligned style. show_turn prints a raw fixed-gutter
    # transcript; this pass tightens the gutter per cell and rewraps so a
    # re-execution reproduces what the PDF expects instead of reverting. Pure
    # layout — content is untouched (asserted) — and idempotent.
    from reflow_transcripts import reflow_notebook
    if reflow_notebook(str(path)):
        print(f"Reflowed: {path.name}")

    if trim:
        from trim_outputs import trim_notebook
        # trim_notebook re-reads the file we just wrote, trims, and writes it
        # back itself. Do NOT nbformat.write(nb, path) afterward — that would
        # clobber the trimmed file on disk with the stale un-trimmed in-memory nb.
        changed = trim_notebook(str(path))
        if changed:
            print(f"Trimmed:  {path.name} ({changed} outputs)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print(__doc__)
        sys.exit(1)
    nb_path = args[0]
    do_trim = "--trim" in args
    execute(nb_path, trim=do_trim)
