"""
trim_outputs.py
After nbconvert execution, cap every text/stdout output at MAX_LINES lines.
Usage: python scripts/trim_outputs.py [notebooks...]
       python scripts/trim_outputs.py chapters/*.ipynb
"""
import json, re, sys, glob
from pathlib import Path

_CHAPTERS = str(Path(__file__).parent.parent / "chapters" / "*.ipynb")

MAX_LINES = 10

_TRIM_MSG_RE = re.compile(r"\.\.\. \[\d+ more lines? trimmed\]\n?$")

def _strip_trim_message(text):
    """Remove a previously-injected trim message so re-runs count correctly."""
    if isinstance(text, str):
        return _TRIM_MSG_RE.sub("", text)
    if isinstance(text, list) and text and _TRIM_MSG_RE.match(text[-1]):
        return text[:-1]
    return text


def trim_output(out: dict) -> dict:
    """Trim a single output object to MAX_LINES lines."""
    # text/plain and stdout are stored as lists of strings (each ending with \n)
    for key in ("text", "data"):
        if key not in out:
            continue
        if key == "text":
            original = out["text"]
            lines = _strip_trim_message(original)
            had_stale = (original != lines)
            if isinstance(lines, list):
                if len(lines) > MAX_LINES:
                    out["text"] = lines[:MAX_LINES] + [f"... [{len(lines) - MAX_LINES} more lines trimmed]\n"]
                elif had_stale:
                    # stale trim message stripped; re-execute notebook for accurate count
                    out["text"] = lines
            elif isinstance(lines, str):
                parts = lines.splitlines(keepends=True)
                if len(parts) > MAX_LINES:
                    out["text"] = "".join(parts[:MAX_LINES]) + f"... [{len(parts) - MAX_LINES} more lines trimmed]\n"
                elif had_stale:
                    # stale trim message stripped; re-execute notebook for accurate count
                    out["text"] = lines
        elif key == "data":
            original_plain = out["data"].get("text/plain", "")
            plain = _strip_trim_message(original_plain)
            had_stale = (original_plain != plain)
            if isinstance(plain, list):
                if len(plain) > MAX_LINES:
                    out["data"]["text/plain"] = plain[:MAX_LINES] + [f"... [{len(plain) - MAX_LINES} more lines trimmed]\n"]
                elif had_stale:
                    out["data"]["text/plain"] = plain
            elif isinstance(plain, str):
                parts = plain.splitlines(keepends=True)
                if len(parts) > MAX_LINES:
                    out["data"]["text/plain"] = "".join(parts[:MAX_LINES]) + f"... [{len(parts) - MAX_LINES} more lines trimmed]\n"
                elif had_stale:
                    out["data"]["text/plain"] = plain
    return out

def merge_stream_outputs(outputs: list) -> list:
    """Merge consecutive stream outputs with the same name into one entry.

    nbconvert captures each sys.stdout.write() call as a separate output
    object, producing hundreds of one-token entries for streaming cells.
    This collapses them so downstream trimming and rendering work correctly.

    Only entries that actually absorbed a sibling are collapsed to a single
    string; a lone stream output is left exactly as it was (list or string),
    so a notebook that needs no merging stays byte-for-byte identical.
    """
    merged = []
    coalesced = []  # parallel flag: did merged[i] absorb a later sibling?
    for out in outputs:
        if (out.get("output_type") == "stream"
                and merged
                and merged[-1].get("output_type") == "stream"
                and merged[-1].get("name") == out.get("name")):
            prev_text = merged[-1]["text"]
            cur_text = out.get("text", [])
            if isinstance(prev_text, list):
                prev_text += cur_text if isinstance(cur_text, list) else [cur_text]
            else:
                prev_text += "".join(cur_text) if isinstance(cur_text, list) else cur_text
            merged[-1]["text"] = prev_text
            coalesced[-1] = True
        else:
            merged.append(out)
            coalesced.append(False)
    # Normalize only the entries we actually merged to a single string, so
    # line-counting in trim_output() sees real newlines rather than token
    # fragments. Untouched stream outputs keep their original representation.
    for out, was_merged in zip(merged, coalesced):
        if was_merged and isinstance(out.get("text"), list):
            out["text"] = "".join(out["text"])
    return merged


def trim_notebook(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    nb = json.loads(raw)
    trimmed = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        if "no-trim" in cell.get("metadata", {}).get("tags", []):
            continue
        cell["outputs"] = merge_stream_outputs(cell.get("outputs", []))
        new_outputs = []
        for out in cell.get("outputs", []):
            before = json.dumps(out)
            out = trim_output(out)
            if json.dumps(out) != before:
                trimmed += 1
            new_outputs.append(out)
        cell["outputs"] = new_outputs
    # Preserve each notebook's on-disk JSON style so untouched cells stay
    # byte-identical: match its unicode escaping (raw unicode vs \u-escaped,
    # detected via isascii) and its trailing-newline convention. Skip the
    # write entirely when nothing changed, so a no-op trim leaves no diff.
    new = json.dumps(nb, indent=1, ensure_ascii=raw.isascii())
    if raw.endswith("\n"):
        new += "\n"
    if new != raw:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
    return trimmed

if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(_CHAPTERS))
    total = 0
    for path in paths:
        n = trim_notebook(path)
        total += n
        if n:
            print(f"  trimmed {n:2d} outputs  {path}")
        else:
            print(f"  ok               {path}")
    print(f"\n{total} outputs trimmed across {len(paths)} notebooks.")
