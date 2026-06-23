"""
lint_code.py
Scan every notebook's code cells and report violations of the code-density rules:
  * Max 15 lines of code per cell (excludes the leading ##system/##user marker line)
  * Max 72 characters per line (excludes the marker line)
  * No semicolon-joined statements on a single line (visual noise + wrap risk)

Outputs a prioritized punch list grouped by chapter, worst first.

Usage:
    uv run python scripts/lint_code.py
    uv run python scripts/lint_code.py chapters/responsible.ipynb
    uv run python scripts/lint_code.py --chapter responsible  # just one chapter
    uv run python scripts/lint_code.py --max 5  # top 5 offenders per chapter
"""
import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_CHAPTERS = str(Path(__file__).parent.parent / "chapters" / "*.ipynb")

MAX_LINES = 15
MAX_CHARS = 72

# Lines starting with these markers are not part of the printed code body.
MARKER_PREFIXES = ("##system", "##user", "##assistant", "##eval", "##model")


@dataclass
class CellReport:
    notebook: str
    cell_index: int
    cell_id: str
    n_lines: int
    over_lines: bool
    long_lines: list[tuple[int, int, str]] = field(default_factory=list)  # (lineno, length, preview)
    semicolon_lines: list[tuple[int, str]] = field(default_factory=list)  # (lineno, preview)

    def severity(self) -> int:
        """Higher = worse. Used to sort the punch list."""
        s = 0
        if self.over_lines:
            s += 100 + (self.n_lines - MAX_LINES) * 5
        s += sum(max(0, ln_len - MAX_CHARS) for _, ln_len, _ in self.long_lines)
        s += len(self.semicolon_lines) * 3
        return s

    def has_issues(self) -> bool:
        return self.over_lines or self.long_lines or self.semicolon_lines


def lint_cell(notebook: str, idx: int, cell: dict) -> CellReport | None:
    raw = cell.get("source", [])
    if isinstance(raw, list):
        src = "".join(raw)
    else:
        src = raw
    lines = src.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    code_lines = [ln for ln in lines if not ln.lstrip().startswith(MARKER_PREFIXES)]
    n = len(code_lines)

    report = CellReport(
        notebook=notebook,
        cell_index=idx,
        cell_id=cell.get("id", "?"),
        n_lines=n,
        over_lines=(n > MAX_LINES),
    )

    for i, ln in enumerate(code_lines, start=1):
        length = len(ln)
        if length > MAX_CHARS:
            report.long_lines.append((i, length, ln.strip()[:80]))
        if _has_semicolon_join(ln):
            report.semicolon_lines.append((i, ln.strip()[:80]))

    return report if report.has_issues() else None


_STRING_RE = re.compile(r"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""")


def _has_semicolon_join(line: str) -> bool:
    """True if a line uses ; to join two real statements (ignoring semicolons inside strings)."""
    no_strings = _STRING_RE.sub("''", line)
    # Strip leading whitespace and comments.
    no_comment = no_strings.split("#", 1)[0]
    return ";" in no_comment.strip().rstrip(";")


def lint_notebook(path: str) -> list[CellReport]:
    with open(path) as f:
        nb = json.load(f)
    out = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        r = lint_cell(path, i, cell)
        if r:
            out.append(r)
    return out


def fmt_report(reports: list[CellReport], max_per_nb: int | None = None) -> str:
    if not reports:
        return "✓ All clear. No code-density violations."
    out = []
    by_nb: dict[str, list[CellReport]] = {}
    for r in reports:
        by_nb.setdefault(r.notebook, []).append(r)
    for nb, rs in sorted(by_nb.items()):
        rs.sort(key=lambda r: -r.severity())
        if max_per_nb:
            rs = rs[:max_per_nb]
        out.append(f"\n━━━ {nb} ({len(rs)} cells with issues) ━━━")
        for r in rs:
            tag_bits = []
            if r.over_lines:
                tag_bits.append(f"{r.n_lines} lines")
            if r.long_lines:
                tag_bits.append(f"{len(r.long_lines)} long lines")
            if r.semicolon_lines:
                tag_bits.append(f"{len(r.semicolon_lines)} semicolons")
            out.append(f"  cell {r.cell_index:3d}  id={r.cell_id}  · {' · '.join(tag_bits)}")
            for ln, length, preview in r.long_lines[:3]:
                out.append(f"      L{ln:2d} [{length} chars]  {preview}")
            for ln, preview in r.semicolon_lines[:3]:
                out.append(f"      L{ln:2d} [;]            {preview}")
            extra = max(0, len(r.long_lines) - 3) + max(0, len(r.semicolon_lines) - 3)
            if extra:
                out.append(f"      … +{extra} more")
    out.append(f"\nTotal cells flagged: {len(reports)}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*", help="Notebook paths or globs")
    p.add_argument("--chapter", help="Just this chapter (basename without .ipynb)")
    p.add_argument("--max", type=int, default=None, help="Max offenders per notebook")
    p.add_argument("--json", action="store_true", help="JSON output")
    args = p.parse_args()

    if args.paths:
        paths = []
        for pat in args.paths:
            paths.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    else:
        paths = sorted(glob.glob(_CHAPTERS))

    if args.chapter:
        paths = [p for p in paths if args.chapter in p]

    all_reports = []
    for p_ in paths:
        all_reports.extend(lint_notebook(p_))

    if args.json:
        print(json.dumps([r.__dict__ for r in all_reports], indent=2))
    else:
        print(fmt_report(all_reports, max_per_nb=args.max))


if __name__ == "__main__":
    main()
