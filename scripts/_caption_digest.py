#!/usr/bin/env python3
"""Digest every figure that will render with a bare \\figlabel (no caption).

For each chapter notebook, finds (a) code cells with image/png outputs and
(b) standalone markdown image paragraphs, then checks whether the next
markdown paragraph is fully italic (the build's caption-absorption rule).
Prints anchor cell ids + surrounding context so captions can be written.
"""
import json
import re
import sys

ORDER = ["front", "intro", "prompting", "tokens", "semantics",
         "metacoding", "augmentation", "agentic", "thinking", "mm", "design",
         "responsible", "efficiency"]

IMG_MD_RE = re.compile(r"^!\[[^\]]*\]\([^)]+\)\s*$")


def first_para(text):
    for p in text.split("\n\n"):
        if p.strip():
            return p.strip()
    return ""


def fully_italic(par):
    par = par.strip()
    return (len(par) > 2 and par.startswith("*") and par.endswith("*")
            and not par.startswith("**") and "\n" not in par.replace("\n", " "))


def src(c):
    return "".join(c["source"])


def heading_before(cells, idx):
    for j in range(idx, -1, -1):
        if cells[j]["cell_type"] == "markdown":
            for line in src(cells[j]).splitlines():
                if line.startswith("#"):
                    h = line.lstrip("#").strip()
            else:
                continue
            return h
    return "?"


def last_heading(cells, idx):
    h = "?"
    for j in range(idx + 1):
        if cells[j]["cell_type"] == "markdown":
            for line in src(cells[j]).splitlines():
                if line.startswith("##"):
                    h = line.lstrip("#").strip()
    return h


def main():
    total = 0
    for name in ORDER:
        nb = json.load(open(f"chapters/{name}.ipynb"))
        cells = nb["cells"]
        found = []
        for i, c in enumerate(cells):
            n_imgs = 0
            if c["cell_type"] == "code":
                for o in c.get("outputs", []):
                    if "image/png" in o.get("data", {}) or "image/jpeg" in o.get("data", {}):
                        n_imgs += 1
                if not n_imgs:
                    continue
                nxt = cells[i + 1] if i + 1 < len(cells) else None
                nxt_para = first_para(src(nxt)) if nxt is not None and nxt["cell_type"] == "markdown" else ""
                if fully_italic(nxt_para):
                    continue
                plot_lines = [l.strip() for l in src(c).splitlines()
                              if re.search(r"plot_|\.show\(|show_|display|imshow|Image", l)][:3]
                found.append((i, c.get("id", "NO-ID"), n_imgs, "code",
                              plot_lines, nxt_para[:260]))
            elif c["cell_type"] == "markdown":
                s = src(c)
                # mm {figure} directives already carry captions; skip
                if "{figure}" in s:
                    continue
                paras = [p for p in s.split("\n\n")]
                for pi, p in enumerate(paras):
                    if IMG_MD_RE.match(p.strip()):
                        nxt_para = ""
                        for q in paras[pi + 1:]:
                            if q.strip():
                                nxt_para = q.strip()
                                break
                        else:
                            if i + 1 < len(cells) and cells[i + 1]["cell_type"] == "markdown":
                                nxt_para = first_para(src(cells[i + 1]))
                        if fully_italic(nxt_para):
                            continue
                        found.append((i, c.get("id", "NO-ID"), 1, "md-img",
                                      [p.strip()[:80]], nxt_para[:260]))
        if found:
            print(f"\n{'='*74}\n{name}.ipynb  ({len(found)} bare figures)")
            for i, cid, n, kind, plot_lines, nxt in found:
                print(f"\n  cell[{i}] id={cid} kind={kind} imgs={n}  section: {last_heading(cells, i)!r}")
                for pl in plot_lines:
                    print(f"      | {pl}")
                print(f"      next-para: {nxt!r}")
            total += len(found)
    print(f"\nTOTAL bare figures: {total}")


if __name__ == "__main__":
    main()
