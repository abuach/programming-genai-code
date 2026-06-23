"""
patch_mystmd.py
Neutralize a mystmd tex-export bug that corrupts hyphens inside code blocks.

myst-to-tex (bundled in mystmd_py/myst.cjs, seen in v1.10.1) serializes every
verbatim block — code cell sources and outputs alike — through its math
escaper, stringToLatexMath. That function inserts a space before any character
it finds in mathReplacements, and the symbols2 table maps the plain ASCII
hyphen to itself. The net effect is that every word-internal hyphen in
rendered code gains a stray space in the PDF: "600-page" becomes "600 -page",
"--tex" becomes "- -tex", "qwen2.5-coder" becomes "qwen2.5 -coder".

Removing the ASCII-hyphen self-mapping fixes verbatim and math output (a raw
"-" is valid in both) without touching prose, which resolves hyphens through
textReplacements first. The unicode minus (−) mapping is kept.

Idempotent: patching twice is a no-op. build_pdf.py calls this before every
myst build, so the fix survives venv reinstalls and mystmd upgrades; if an
upgrade ships a bundle where the pattern no longer exists (e.g. fixed
upstream), this reports and changes nothing.

Usage: python scripts/patch_mystmd.py
"""
import sys
from pathlib import Path

BUGGY = (
    '  "\\u2212": "-",\n'
    "  // minus\n"
    '  "-": "-",\n'
    "  // hyphen minus\n"
)
FIXED = (
    '  "\\u2212": "-",\n'
    "  // minus\n"
)


def find_bundle() -> Path:
    import mystmd_py
    return Path(mystmd_py.__file__).parent / "myst.cjs"


def patch() -> bool:
    """Apply the patch if needed. Returns True if the bundle is in the fixed
    state on exit (already patched or patched now), False if the pattern was
    not found at all."""
    bundle = find_bundle()
    raw = bundle.read_text(encoding="utf-8")
    if BUGGY not in raw:
        # Either already patched, or a new mystmd whose bundle changed shape.
        already = '"-": "-"' not in raw
        print(f"  mystmd hyphen patch: nothing to do "
              f"({'already patched' if already else 'pattern not found — check mystmd version'})")
        return already
    # The same table appears in the tex and typst serializers; fix both.
    bundle.write_text(raw.replace(BUGGY, FIXED), encoding="utf-8")
    print(f"  mystmd hyphen patch: applied to {bundle}")
    return True


if __name__ == "__main__":
    sys.exit(0 if patch() else 1)
