#!/usr/bin/env python3
"""Inventory every figure in the book and emit an alt-text authoring scaffold.

Walks each chapter notebook in myst.yml TOC order, finds every figure-producing
unit in document order (code-cell image outputs, standalone markdown images,
MyST {figure} directives), and records for each a 1-based per-chapter ordinal,
a source-code/path hint, and a best-effort caption so a human can verify the
ordinal maps to the right figure.

Cross-checks each chapter's unit count against the number of \\includegraphics
lines in the emitted exports/programming-genai-<chapter>.tex. If those agree,
alt text keyed by (chapter, ordinal) can be injected by walking images in
document order at build time.

Run:  uv run python scripts/_alt_scaffold.py
Writes the scaffold to scripts/alt_texts.yml (only fills entries that don't
exist yet; never clobbers authored `alt:` text).
"""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
EXPORTS = ROOT / "exports"
SCAFFOLD = ROOT / "scripts" / "alt_texts.yml"

# Chapter notebooks in myst.yml TOC order (glossary/cover/toc carry no figures).
CHAPTERS = ["front", "intro", "prompting", "tokens", "semantics",
            "metacoding", "augmentation", "agentic", "autonomy", "thinking",
            "mm", "efficiency", "responsible"]

IMG_MD_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
FIG_DIR_RE = re.compile(r"^```\{figure\}\s*(?P<path>\S+)", re.M)


def src(cell):
    return "".join(cell["source"])


def italic_caption(par):
    """Return inner text if `par` is a single fully-italic line (the book's
    caption convention), else None. Mirrors build_pdf's _full_textit rule."""
    p = par.strip()
    if len(p) > 2 and p.startswith("*") and not p.startswith("**") and p.endswith("*"):
        return p[1:-1].strip()
    return None


def first_para(text):
    for p in text.split("\n\n"):
        if p.strip():
            return p.strip()
    return ""


def next_caption(cells, i):
    """Best-effort caption following cell i: first paragraph of the next
    markdown cell, if it is fully italic."""
    nxt = cells[i + 1] if i + 1 < len(cells) else None
    if nxt and nxt["cell_type"] == "markdown":
        return italic_caption(first_para(src(nxt)))
    return None


def code_hint(cell):
    lines = [l.strip() for l in src(cell).splitlines()
             if re.search(r"plot_|\.show\(|show_|display\(|imshow|Image|savefig|plt\.", l)]
    return " | ".join(lines[:2]) if lines else "(code cell, no plot call matched)"


def units_for_chapter(name):
    """Figure-producing units in document order for one chapter notebook."""
    nb = json.loads((ROOT / "chapters" / f"{name}.ipynb").read_text())
    cells = nb["cells"]
    units = []
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            imgs = [o for o in c.get("outputs", [])
                    if "image/png" in o.get("data", {}) or "image/jpeg" in o.get("data", {})]
            for k, _ in enumerate(imgs):
                cap = next_caption(cells, i) if k == len(imgs) - 1 else None
                units.append({"source": "code", "hint": code_hint(c),
                              "caption": cap, "existing_alt": ""})
        elif c["cell_type"] == "markdown":
            text = src(c)
            for m in FIG_DIR_RE.finditer(text):
                units.append({"source": "figure", "hint": m.group("path"),
                              "caption": None, "existing_alt": ""})
            for line in text.splitlines():
                mm = IMG_MD_RE.search(line)
                if mm and line.strip().startswith("!["):
                    units.append({"source": "image", "hint": mm.group("path"),
                                  "caption": next_caption(cells, i),
                                  "existing_alt": mm.group("alt")})
    return units


def tex_image_count(name):
    f = EXPORTS / f"programming-genai-{name}.tex"
    if not f.exists():
        return None
    return f.read_text().count(r"\includegraphics")


def main():
    existing = {}
    if SCAFFOLD.exists():
        existing = yaml.safe_load(SCAFFOLD.read_text()) or {}

    scaffold = {}
    grand_nb = grand_tex = 0
    print(f"{'chapter':<14}{'notebook':>9}{'tex':>6}   status")
    print("-" * 46)
    for name in CHAPTERS:
        units = units_for_chapter(name)
        tex_n = tex_image_count(name)
        nb_n = len(units)
        grand_nb += nb_n
        grand_tex += tex_n or 0
        status = "PASS" if tex_n == nb_n else f"MISMATCH (tex {tex_n} vs nb {nb_n})"
        print(f"{name:<14}{nb_n:>9}{str(tex_n):>6}   {status}")

        prior = {e["ordinal"]: e for e in existing.get(name, [])}
        entries = []
        for k, u in enumerate(units, 1):
            authored = prior.get(k, {}).get("alt", "")
            entries.append({
                "ordinal": k,
                "source": u["source"],
                "hint": u["hint"],
                "caption": u["caption"] or "",
                "existing_alt": u["existing_alt"],
                "alt": authored,
            })
        scaffold[name] = entries

    print("-" * 46)
    print(f"{'TOTAL':<14}{grand_nb:>9}{grand_tex:>6}")

    class _LiteralDumper(yaml.SafeDumper):
        pass

    def _str_rep(dumper, data):
        style = "|" if "\n" in data else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)
    _LiteralDumper.add_representer(str, _str_rep)

    SCAFFOLD.write_text(yaml.dump(scaffold, Dumper=_LiteralDumper, sort_keys=False,
                                  allow_unicode=True, width=100))
    print(f"\nwrote {SCAFFOLD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
