#!/usr/bin/env python3
"""
build_submission.py — Assemble a Springer Nature book submission package.

Springer's "Submitting your manuscript" guide asks for three folders:

  source/       all LaTeX sources of the final manuscript, plus the style/
                fonts/figures/bibliography they need, self-contained.
  pdf/          a PDF of the whole manuscript and one PDF per chapter, all
                fonts embedded.
  permissions/  third-party permissions.

This script packages the artifacts already produced in exports/ by
scripts/build_pdf.py into submission/. It does not compile anything itself:
run `python scripts/build_pdf.py --all` first (or pass --build) so the PDFs and
the generated tex are current, then run this.

The submission/source/ tree is self-contained and compiles on its own — the
generated tex points at ../fonts/ and ../images/ (one level up from exports/),
which we localise to fonts/ and images/ sitting beside the tex.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXPORTS = ROOT / "exports"
FONTS = ROOT / "fonts"
COVER = ROOT / "images" / "bookcover.png"
CHAPTERS = ROOT / "chapters"
SUBMISSION = ROOT / "submission"

MAIN_TEX = EXPORTS / "programming-genai.tex"
MAIN_BIB = EXPORTS / "main.bib"
BOOK_PDF = EXPORTS / "programming-genai.pdf"
CHAPTER_PDF_DIR = EXPORTS / "chapters"


# ---------------------------------------------------------------------------
# source/
# ---------------------------------------------------------------------------

def _live_chapter_tex():
    r"""Return the chapter tex files actually \include'd by the main document.

    exports/ also holds stale tex from removed chapters (design, foundations)
    and structural stubs (citation, frontmatter); parsing the include list keeps
    those out of the submission so only the real manuscript ships."""
    tex = MAIN_TEX.read_text(encoding="utf-8")
    names = re.findall(r"\\include\{(programming-genai-[a-z]+)\}", tex)
    return [EXPORTS / f"{n}.tex" for n in names]


def _localise_paths(text: str) -> str:
    r"""Make the bundle self-contained: the generated tex loads the body font
    from ../fonts/ and the cover from ../images/ (relative to the exports/
    compile dir). Inside source/ those dirs sit beside the tex, so drop the ../."""
    return text.replace("../fonts/", "fonts/").replace("../images/", "images/")


def assemble_source(dest: Path):
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # Main tex + every live chapter tex, with the two parent-relative asset
    # paths localised so the tree compiles in place.
    tex_files = [MAIN_TEX, *_live_chapter_tex()]
    for src in tex_files:
        (dest / src.name).write_text(
            _localise_paths(src.read_text(encoding="utf-8")), encoding="utf-8")

    # Bibliography named by \addbibresource{main.bib}.
    shutil.copy2(MAIN_BIB, dest / MAIN_BIB.name)

    # Bundled body font (EB Garamond instances) + its OFL licence and the
    # provenance README — the licence must travel with the fonts.
    shutil.copytree(FONTS, dest / "fonts")

    # The cover is the only root image the tex references; the 294 chapter
    # figures live in exports/files/ and are copied below.
    (dest / "images").mkdir()
    shutil.copy2(COVER, dest / "images" / COVER.name)

    shutil.copytree(EXPORTS / "files", dest / "files")

    n_fig = sum(1 for p in (dest / "files").iterdir() if p.is_file())
    print(f"  source/: {len(tex_files)} tex files, main.bib, "
          f"{n_fig} figures, fonts/, cover")


# ---------------------------------------------------------------------------
# pdf/
# ---------------------------------------------------------------------------

def assemble_pdf(dest: Path):
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    if BOOK_PDF.exists():
        shutil.copy2(BOOK_PDF, dest / BOOK_PDF.name)
    else:
        print("  WARNING: exports/programming-genai.pdf missing — run "
              "build_pdf.py (no flag or --all) for the whole-manuscript PDF")

    chap_dest = dest / "chapters"
    chap_dest.mkdir()
    pdfs = sorted(CHAPTER_PDF_DIR.glob("*.pdf")) if CHAPTER_PDF_DIR.exists() else []
    for p in pdfs:
        shutil.copy2(p, chap_dest / p.name)
    if not pdfs:
        print("  WARNING: no per-chapter PDFs — run build_pdf.py --chapters")

    have_book = (dest / BOOK_PDF.name).exists()
    print(f"  pdf/: {'whole manuscript + ' if have_book else ''}"
          f"{len(pdfs)} chapter PDF(s)")


# ---------------------------------------------------------------------------
# permissions/
# ---------------------------------------------------------------------------
#
# Third-party permissions for this book are the licences of the photographs
# downloaded from the web (the multimodal chapter's real-world images). They are
# recorded in LICENSE(S).json files next to the images. We copy those raw records
# in and generate a human-readable credits manifest filtered to the images that
# actually appear in the manuscript. The editor's written AI-image approval and
# any other permission documents are author-supplied; we never delete files we
# did not generate, so dropping them into permissions/ is safe across reruns.

PERMISSIONS_README = """\
Permissions
===========

This folder holds the third-party permissions for the manuscript.

  THIRD-PARTY-IMAGE-CREDITS.txt
      Generated credits for every web-sourced image used in the book, with
      title, author, year, licence, and source URL. Public-domain images need
      no permission (source documented); CC BY-SA images need attribution and
      their share-alike terms confirmed with the editor.

  AI-IMAGE-GENERATION.txt
      The model and licence behind the book's AI-generated illustrations.

  image-licenses/
      The raw LICENSE(S).json records copied verbatim from the repository.

Author-supplied documents to add here before submission:

  * The editor's written approval for the AI-generated images (Thomas
    Hempfling, under the publisher's "book about AI" exception, covering the
    in-text demonstrations and the chapter openers). See AI-IMAGE-GENERATION.txt
    for the model licence that also permits this use.

  * Any permission letters for third-party figures, tables, or long excerpts
    added later.

matplotlib charts, code listings, and transcripts are the author's own work
and need no third-party permission.
"""

# The book's only image-generation model. FLUX.1-schnell ships under Apache 2.0,
# the commercially-usable variant (FLUX.1-dev is non-commercial), so it permits
# both the model's use and commercial use of the images it generates — which,
# alongside the editor's written approval, is what clears the AI illustrations
# for a paid book. Confirmed against the references.bib `flux2024schnell` entry.
AI_IMAGE_GENERATION = """\
AI-generated images
===================

The book's illustrations -- the chapter opener images and the in-text image
demonstrations -- were generated by the author with an open-weights text-to-image
model. They are the author's own outputs, not third-party works, and are
additionally covered by the editor's written approval (see README.txt). Each one
is labelled "Image generated by AI (FLUX.1-schnell)" in its caption.

Model
  Name      FLUX.1-schnell
  Author    Black Forest Labs
  Year      2024
  Licence   Apache License 2.0
  Source    https://huggingface.co/black-forest-labs/FLUX.1-schnell
  Citation  flux2024schnell (references.bib)

Licence note
  FLUX.1-schnell is released under the Apache 2.0 licence, which permits
  commercial use and places no restriction on the images it generates. It is the
  open, commercially-usable release; the FLUX.1-dev and FLUX.1-pro variants are
  not, and are not used in this book. No other image-generation model is used.
"""

# A licence record entry describes a third-party (web-sourced) image when its
# source is a URL or its licence names a real licence (Creative Commons / Public
# Domain). The author's own generated figures carry "Original work" licences and
# a "Generated with ..." source, so they fall through.
_THIRD_PARTY_LICENCE_RE = re.compile(r"\bCC[ -]|Creative Commons|Public Domain",
                                     re.IGNORECASE)


def _is_third_party(entry: dict) -> bool:
    src = str(entry.get("source", ""))
    lic = str(entry.get("license", ""))
    return src.startswith("http") or bool(_THIRD_PARTY_LICENCE_RE.search(lic))


def _image_used(chapter_slug: str, filename: str) -> bool:
    """True if ``filename`` is referenced in chapters/<slug>.ipynb (the notebook
    loads its images by path, so a basename match is a reliable usage signal).
    Unknown (no such notebook) counts as used so nothing is silently dropped."""
    nb = CHAPTERS / f"{chapter_slug}.ipynb"
    if not nb.exists():
        return True
    return filename in nb.read_text(encoding="utf-8")


def _find_license_files():
    """Every LICENSE(S).json under chapters/images/ and images/, in sorted order."""
    found = []
    for base in (CHAPTERS / "images", ROOT / "images"):
        if base.exists():
            for pat in ("LICENSE.json", "LICENSES.json"):
                found.extend(base.rglob(pat))
    return sorted(set(found))


def assemble_permissions(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    dest.joinpath("README.txt").write_text(PERMISSIONS_README, encoding="utf-8")
    dest.joinpath("AI-IMAGE-GENERATION.txt").write_text(
        AI_IMAGE_GENERATION, encoding="utf-8")

    license_files = _find_license_files()
    raw_dir = dest / "image-licenses"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir()

    sections, skipped, listed = [], [], []
    counts = {"files": 0}
    for lf in license_files:
        chapter = lf.parent.name                       # chapters/images/<chapter>/
        shutil.copy2(lf, raw_dir / f"{chapter}-{lf.name}")
        counts["files"] += 1
        try:
            records = json.loads(lf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        lines = [f"{chapter}  ({lf.relative_to(ROOT)})",
                 "-" * 60]
        used_here = 0
        for fname, e in records.items():
            if not _is_third_party(e):
                continue                                # author's own generated work
            if not _image_used(chapter, fname):
                skipped.append(f"{chapter}/{fname}")
                continue
            used_here += 1
            listed.append((fname, str(e.get("license", ""))))
            lines += [
                f"  {fname}",
                f"    Title:   {e.get('title','')}",
                f"    Author:  {e.get('author','')}",
                f"    Year:    {e.get('year','')}",
                f"    Licence: {e.get('license','')}",
                f"    Source:  {e.get('source','')}",
            ]
        if used_here:
            sections.append("\n".join(lines))

    # Licence summary: group by licence, flag the ones that need editor action
    # (share-alike and non-commercial terms can clash with a commercial book).
    by_licence = {}
    for fname, lic in listed:
        by_licence.setdefault(lic, []).append(fname)
    summary = ["Licences in use", "-" * 60]
    for lic in sorted(by_licence):
        flags = []
        if re.search(r"BY-SA|ShareAlike", lic, re.IGNORECASE):
            flags.append("share-alike")
        if re.search(r"BY-NC|NonCommercial", lic, re.IGNORECASE):
            flags.append("non-commercial")
        note = f"   [{' + '.join(flags)} — confirm with editor]" if flags else ""
        summary.append(f"  {lic}{note}")
        summary.append(f"      {', '.join(sorted(by_licence[lic]))}")

    credits = (
        "Third-party image credits\n"
        "=========================\n\n"
        "Figures reproduced from third-party sources, with the licence on record\n"
        "for each. Images the author created (matplotlib charts, generated\n"
        "diagrams, AI illustrations) are original work and are not listed here;\n"
        "the AI illustrations are covered by the editor's written approval\n"
        "(see README.txt).\n\n"
        + ("\n\n".join(sections) + "\n\n" + "\n".join(summary) if sections
           else "No third-party web-sourced images are used in the manuscript.")
        + "\n"
    )
    dest.joinpath("THIRD-PARTY-IMAGE-CREDITS.txt").write_text(credits, encoding="utf-8")

    print(f"  permissions/: README + credits for {len(listed)} third-party "
          f"image(s) from {counts['files']} licence file(s)")
    if skipped:
        print(f"    note: {len(skipped)} licensed image(s) not used in the "
              f"manuscript, omitted from credits: {', '.join(skipped)}")


# ---------------------------------------------------------------------------
# top-level README
# ---------------------------------------------------------------------------

TOP_README = """\
Programming Generative AI — Springer manuscript submission package
Author: Chiké Abuah

Layout follows Springer Nature's "Submitting your manuscript" guidelines:

  source/       LaTeX source of the final manuscript, self-contained.
  pdf/          PDF of the whole manuscript (programming-genai.pdf) and one
                PDF per chapter (pdf/chapters/Abuah-Chap1.pdf ...), all fonts
                embedded.
  permissions/  Third-party permissions (see permissions/README.txt).

COMPILING source/
-----------------
  Engine     XeLaTeX is required — both fonts are loaded with fontspec from
             source/fonts/ by path: EB Garamond (body) and IBM Plex Mono
             (code/output). No system font installation is needed.
  References biber (biblatex; each chapter prints its own reference list).
  Code       the minted package, which needs shell-escape enabled and Python
             Pygments (pip install Pygments) on the build machine.

  Build with latexmk:

      cd source
      latexmk -xelatex -shell-escape -e '$max_repeat=9' programming-genai.tex

  The extra passes let the table of contents, lists of figures/tables, index,
  and per-chapter bibliographies converge.

The canonical rendering is pdf/programming-genai.pdf.
"""


def write_top_readme():
    (SUBMISSION / "README.txt").write_text(TOP_README, encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_build():
    """Refresh exports/ via build_pdf.py --all (whole book + per-chapter PDFs)."""
    print("\n── Refreshing exports/ (build_pdf.py --all) ──────────────")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("build_pdf.py")), "--all"],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Assemble the Springer submission package in submission/.")
    parser.add_argument(
        "--build", action="store_true",
        help="run build_pdf.py --all first so the PDFs and tex are current")
    args = parser.parse_args()

    if args.build:
        run_build()

    if not MAIN_TEX.exists():
        print(f"ERROR: {MAIN_TEX} not found — run build_pdf.py first "
              "(or pass --build).", file=sys.stderr)
        sys.exit(1)

    print("\n── Assembling submission/ ────────────────────────────────")
    SUBMISSION.mkdir(exist_ok=True)
    assemble_source(SUBMISSION / "source")
    assemble_pdf(SUBMISSION / "pdf")
    assemble_permissions(SUBMISSION / "permissions")
    write_top_readme()

    total_mb = sum(
        p.stat().st_size for p in SUBMISSION.rglob("*") if p.is_file()
    ) // (1024 * 1024)
    print(f"\n✓ submission/ assembled ({total_mb} MB)\n  {SUBMISSION}")


if __name__ == "__main__":
    main()
