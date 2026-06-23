#!/usr/bin/env python3
"""reflow_transcripts.py — re-wrap saved show_turn transcripts to a fixed width.

The labelled transcripts printed by ``genai.agent.show_turn`` right-align a
speaker into a 14-char gutter, then ``" | "``, then the wrapped body:

    SOPHIA | Your new GPA will be approximately 3.34. Attention is the
           | relevance weight assigned to every input token per output

Several notebooks carry transcript *outputs* that were saved when ``show_turn``
wrapped to a wider budget than it does today. Those stale lines overflow the
PDF output box and LaTeX wraps them a second time into a ``↪`` staircase that
breaks the gutter alignment. Re-executing is not an option for frozen cells and
would drift the model text for the rest, so instead we re-flow the *existing*
saved text: reconstruct each turn's body from its wrapped lines and re-wrap it
at the target budget. Pure layout — not one word of content changes.

Usage:
    reflow_transcripts.py <notebook> [--budget N] [--apply]
    reflow_transcripts.py <notebook> --cells 18,22 --apply

Without --apply it is a dry run: it reports, per affected cell, the old/new
line counts and longest line, and asserts the reflow loses no content.
Byte-stability follows the project rule (detect ensure_ascii + trailing newline
from the original bytes; skip the write when nothing changed).
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

# A transcript line is ``<gutter><' | '><body>`` where the gutter holds a
# right- or left-padded speaker label (or only spaces, on a continuation line).
# The gutter width varies per cell — show_turn used a fixed 14, the auto-gutter
# output uses each cell's longest label, and other helpers (e.g. show_judge) use
# their own — so we locate the separator dynamically rather than assume a width.
# This also makes the reflow idempotent: it can re-parse its own output.
_SEP = " | "
_MAXGUT = 14        # the separator must fall within the left gutter region


def _sep_col(line: str) -> int:
    """Column of the gutter ``| `` separator, or -1 if this isn't a turn line."""
    j = line.find(_SEP)
    return j if 0 <= j <= _MAXGUT else -1


def _is_transcript_line(line: str) -> bool:
    return _sep_col(line) != -1


def _speaker(line: str) -> str:
    """The speaker label, or '' on a continuation line (all-space gutter)."""
    j = _sep_col(line)
    return line[:j].strip() if j != -1 else ""


def _body(line: str) -> str:
    """The text after the gutter separator."""
    return line[_sep_col(line) + len(_SEP):]


def _parse_turns(lines, start):
    """From ``lines[start]`` (a turn-start), consume one contiguous transcript
    block. Returns ``(turns, next_index)`` where turns is a list of
    ``(speaker, body_lines)`` — the body's *original* lines, kept separate.

    Continuation lines are NOT joined: a turn's body may carry intentional
    structure (a code snippet's newlines and indentation, a hard-wrapped
    paragraph), and joining then re-wrapping would flatten it. Preserving the
    lines keeps that structure and makes the reflow idempotent."""
    turns = []
    i = start
    while i < len(lines) and _is_transcript_line(lines[i]) and _speaker(lines[i]) != "":
        speaker = _speaker(lines[i])
        body_lines = [_body(lines[i])]
        i += 1
        while i < len(lines) and _is_transcript_line(lines[i]) and _speaker(lines[i]) == "":
            body_lines.append(_body(lines[i]))
            i += 1
        turns.append((speaker, body_lines))
    return turns, i


def _is_pure_wrap(body_lines) -> bool:
    """Whether a turn's body is plain text that was merely word-wrapped (so it
    can be safely re-joined and re-wrapped to a new width), as opposed to text
    carrying intentional line structure that must be preserved verbatim.

    Preserve when a line is indented (a code snippet) or when re-wrapping the
    joined text at its own widest line fails to reproduce the stored lines —
    that mismatch means a break was deliberate (a tool signature above its
    description, a ``` fence, a parenthetical on its own line), not a wrap."""
    if len(body_lines) < 2:
        return True
    if any(bl[:1].isspace() for bl in body_lines):
        return False
    # Compare against whitespace-normalized lines: a prose body may carry an
    # incidental double space that textwrap collapses, which would otherwise
    # make this test (and so the reflow) unstable across passes.
    norm = [" ".join(bl.split()) for bl in body_lines]
    width = max(len(bl) for bl in norm)
    return textwrap.wrap(" ".join(norm), width=width,
                         break_on_hyphens=False) == norm


def reflow_text(text: str, budget: int, gutter="auto", align="right") -> str:
    """Re-gutter every transcript block in ``text`` so it fits ``budget`` columns.

    Non-transcript lines pass through untouched. Each contiguous block (two or
    more turn lines) is re-emitted with a new gutter: the block's longest label
    when ``gutter='auto'`` (so short-label transcripts stop wasting a wide left
    margin) or a fixed int, with the label padded left/right per ``align``.

    A plain word-wrapped body (``_is_pure_wrap``) is re-joined and re-wrapped to
    the new, wider body width — fixing the cramped breaks the old narrow gutter
    forced. A structured body (code, fences, deliberate breaks) is kept line for
    line, splitting only a line that still overflows. Both paths are idempotent."""
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        if not _is_transcript_line(lines[i]) or _speaker(lines[i]) == "":
            out.append(lines[i])
            i += 1
            continue
        turns, j = _parse_turns(lines, i)
        # A lone gutter line (e.g. a one-off "QUESTION | ..." header above
        # free-form text) isn't a transcript to re-gutter; leave it verbatim.
        if sum(len(bl) for _, bl in turns) < 2:
            out.append(lines[i])
            i += 1
            continue
        g = max(len(s) for s, _ in turns) if gutter == "auto" else int(gutter)
        width = budget - g - len(_SEP)
        pad = (lambda s: f"{s:<{g}}") if align == "left" else (lambda s: f"{s:>{g}}")
        cont = pad("") + _SEP
        for speaker, body_lines in turns:
            if len(body_lines) == 1:
                # A single body line carries no wrap to redo; keep it verbatim
                # (preserving any intentional spacing) and split only if it now
                # overflows the box.
                bl = body_lines[0]
                emitted = ([bl] if len(bl) <= width
                           else textwrap.wrap(bl, width=width,
                                              break_on_hyphens=False) or [""])
            elif _is_pure_wrap(body_lines):
                emitted = textwrap.wrap(" ".join(" ".join(body_lines).split()),
                                        width=width,
                                        break_on_hyphens=False) or [""]
            else:
                emitted = []
                for bl in body_lines:
                    emitted.extend([bl] if len(bl) <= width
                                   else textwrap.wrap(bl, width=width,
                                              break_on_hyphens=False) or [""])
            out.append(pad(speaker) + _SEP + emitted[0])
            out.extend(cont + e for e in emitted[1:])
        i = j
    return "\n".join(out)


def _content_tokens(text: str) -> list:
    """Body tokens of every transcript line, for a content-preservation check."""
    toks = []
    for line in text.split("\n"):
        if _is_transcript_line(line):
            toks.extend(_body(line).split())
    return toks


def _max_len(text: str) -> int:
    return max((len(l) for l in text.split("\n")), default=0)


def _line_count(text: str) -> int:
    return len([l for l in text.split("\n") if l.strip()])


def process_notebook(path: Path, budget: int, apply: bool, only_cells,
                     gutter="auto", align="right"):
    raw = path.read_bytes()
    nb = json.loads(raw)
    changed = False
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if only_cells is not None and idx not in only_cells:
            continue
        for out in cell.get("outputs", []):
            # stream output uses "text"; execute_result/display_data use data[text/plain]
            holder, key = (out, "text") if "text" in out else (
                out.get("data", {}), "text/plain")
            val = holder.get(key)
            if val is None:
                continue
            original = "".join(val) if isinstance(val, list) else val
            if "\n" not in original or _SEP not in original:
                continue
            reflowed = reflow_text(original, budget, gutter, align)
            if reflowed == original:
                continue
            # content-preservation guard: not one word may be added or lost
            assert _content_tokens(reflowed) == _content_tokens(original), (
                f"cell {idx}: reflow changed content")
            print(f"  cell {idx}: lines {_line_count(original)}->{_line_count(reflowed)}"
                  f"  maxlen {_max_len(original)}->{_max_len(reflowed)}"
                  + ("  [>10 LINES]" if _line_count(reflowed) > 10 else ""))
            if apply:
                holder[key] = reflowed.splitlines(keepends=True) if isinstance(val, list) else reflowed
                changed = True

    if apply and changed:
        ensure_ascii = raw.isascii()
        text = json.dumps(nb, indent=1, ensure_ascii=ensure_ascii)
        if raw.endswith(b"\n"):
            text += "\n"
        new_bytes = text.encode("utf-8")
        if new_bytes != raw:
            path.write_bytes(new_bytes)
            print(f"  wrote {path.name}")
    return changed


def reflow_notebook(path, budget: int = 69, gutter="auto", align="left") -> bool:
    """Reflow every transcript block in a notebook in place; return True if the
    file changed. This is the entry point execute_notebook.py calls after a run
    so re-executing a chapter reproduces the book's auto-gutter transcript style
    instead of reverting to the raw fixed-gutter show_turn output."""
    return process_notebook(Path(path), budget, True, None, gutter, align)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--budget", type=int, default=69,
                    help="max characters per rendered line (gutter + body)")
    ap.add_argument("--cells", default=None,
                    help="comma-separated cell indices to limit to")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default: dry run)")
    ap.add_argument("--gutter", default="auto",
                    help="'auto' (per-block max label) or a fixed int")
    ap.add_argument("--align", default="right", choices=["left", "right"],
                    help="pad the speaker label left or right of the gutter")
    args = ap.parse_args()
    only = None if args.cells is None else {int(c) for c in args.cells.split(",")}
    print(f"{args.notebook.name}  budget={args.budget}  gutter={args.gutter}  "
          f"align={args.align}  {'APPLY' if args.apply else 'dry-run'}")
    process_notebook(args.notebook, args.budget, args.apply, only,
                     args.gutter, args.align)


if __name__ == "__main__":
    main()
