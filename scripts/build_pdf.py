#!/usr/bin/env python3
"""
build_pdf.py — Post-processing PDF build pipeline for Programming GenAI.

Pipeline:
  1. mystmd build --tex
  2. Convert verbatim blocks → minted environments
  3. latexmk compile with XeLaTeX + shell escape

Requirements:
  - uv
  - mystmd
  - MacTeX / TeX Live (latexmk + xelatex)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXPORTS = ROOT / "exports"

# MyST emits every matplotlib cell-output figure at a fixed `width=0.7\linewidth`
# regardless of the figure's own size. On a letterpaper page with 1.25in side
# margins the text block is 6in wide, so 0.7 leaves charts at 4.2in with ~1.8in
# of dead margin — small enough that chart labels become hard to read in print.
# We rewrite that default to FIGURE_WIDTH so charts fill more of the page. Photos
# and hand-sized diagrams carry their own explicit widths (0.55/0.65/0.8) and are
# left untouched. Dial this toward 1.0 for full-bleed charts; 0.95 keeps a thin
# breathing margin. A square figure at 0.95 is ~5.7in tall, well within the 9in
# text height, so this is safe for every aspect ratio currently in the book.
MYST_DEFAULT_FIGURE_WIDTH = "0.7"
FIGURE_WIDTH = "0.95"

# MyST rewrites Unicode math glyphs inside code/output as LaTeX macros. In a
# verbatim block those must be literal characters, so we map each macro back to
# its glyph before emitting minted. XeLaTeX + IBM Plex Mono render these fine
# (the long-standing \rightarrow case proves the path). Add new symbols here
# rather than replacing the glyph with ASCII in the notebook source.
UNICODE_MACRO_RESTORE = {
    r"\leftrightarrow": "↔",
    r"\Leftrightarrow": "⇔",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\Rightarrow": "⇒",
    r"\Leftarrow": "⇐",
    r"\times": "×",
    r"\approx": "≈",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\neq": "≠",
}

# ---------------------------------------------------------------------------
# Heuristics for detecting Python code blocks
# ---------------------------------------------------------------------------

PYTHON_RE = re.compile(
    r"^\s*(import |from [\w.]+\s+import|def |class |for |if |elif |else:|while |"
    r"with |try:|except[\s:(]|return |assert |raise |yield |"
    r"# [^\n]|@\w|lambda )",   # "# " not "##"; "@word" not "@mention"
    re.MULTILINE,
)

ASSIGNMENT_RE = re.compile(
    r"^\s*\w[\w.]*\s*=\s*\S",
    re.MULTILINE,
)

# The book's smallest code cells are nothing but bare helper calls
# (`plot_scoreboard(...)`, `show_memory(stream)`) — no import, keyword, or
# assignment line for the other regexes to hit. The second alternative
# matches a whole line that is one call. Output blocks can't be flipped by a
# stray `word(...)` line in a response: replace_verbatim forces every output
# environment to text regardless of what is_python returns.
CALL_RE = re.compile(
    r"^\s*(print|len|range|enumerate|zip|map|filter)\(|"
    r"^\s*[A-Za-z_][\w.]*\(.*\)\s*$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def run(cmd, **kwargs):
    """Run a normal system command."""

    print(f"\n$ {' '.join(cmd)}")

    result = subprocess.run(cmd, **kwargs)

    # latexmk:
    #   0  = success
    #   1  = warnings only
    #   12 = warnings only
    if result.returncode not in (0, 1, 12):
        print(
            f"ERROR: exit code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    return result


def run_python_module(module, args, **kwargs):
    """
    Run a Python module through uv-managed Python.

    Equivalent to:
        uv run python -m <module> ...
    """

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        module,
        *args,
    ]

    print(f"\n$ {' '.join(cmd)}")

    result = subprocess.run(cmd, **kwargs)

    if result.returncode not in (0, 1, 12):
        print(
            f"ERROR: exit code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    return result


# ---------------------------------------------------------------------------
# Verbatim → Minted conversion
# ---------------------------------------------------------------------------

# Magic comments placed as the first line of a notebook code cell.
# Colors (defined in templates/plain_latex/template.tex): code = steel blue,
# model output = ochre, results = slate, editorial = teal.
# ##system  -> steel systemcell;   following output -> slate sysoutcell
# ##user    -> steel usercodecell; following output -> ochre modeloutcell
# ##model   -> ochre modeloutcell; for explicitly displaying stored model responses
# ##eval    -> steel usercodecell; following output -> slate evaloutcell (merged w/ results)
CELL_MARKERS = {
    "##system": ("systemcell",   "sysoutcell"),
    "##user":   ("usercodecell", "modeloutcell"),
    "##model":  ("modeloutcell", None),
    "##eval":   ("usercodecell", "evaloutcell"),
}

# Minted options for blocks inside a semantic tcolorbox (no double-border).
# Code cells always use a white background so syntax highlighting reads cleanly.
_INLINE_PYTHON_OPTS = (
    "frame=none,xleftmargin=18pt,linenos=true,numbersep=8pt,"
    "bgcolor=white,fontsize=\\small,breaklines=true,breakanywhere=true,tabsize=4"
)

# Output/response cells use the box's own tinted background so they are
# visually distinct from the code above them.  Font is handled globally by
# \setminted[text]{fontfamily=ibmplexmono} in the template (IBM Plex Mono when
# available, default tt otherwise) — we deliberately do not override it here.
_OUTPUT_BG = {
    "modeloutcell": "modelbg",   # ochre cream — model response
    "evaloutcell":  "evalbg",    # slate tint  — scores / metrics (merged w/ results)
    "sysoutcell":   "databg",    # slate tint  — program / infrastructure output
    "systemcell":   "sysbg",     # steel tint  — system code (rare text output)
    "examplecell":  "white",     # white       — author-drawn editorial example
}

def _inline_text_opts(cell_env: str, transcript: bool = False) -> str:
    bg = _OUTPUT_BG.get(cell_env, "outputbg")
    if transcript:
        # Labelled transcripts (genai.show_turn) are pre-wrapped to fit the box,
        # so LaTeX never needs to break them. With breaklines on, fancyvrb nudges
        # near-full lines ~1 char left and the gutter pipes stop aligning; with it
        # off the leading spaces are emitted verbatim and every pipe lines up
        # exactly. Safe only because show_turn caps each line under the box width.
        return (f"frame=none,xleftmargin=0pt,bgcolor={bg},"
                r"fontsize=\small,breaklines=false")
    # Other text output (dict dumps, tracebacks, long URLs) still needs wrapping.
    # breakautoindent=false: wrapped lines fall to a fixed indent instead of the
    # broken line's own column, so a wrap reads as flow rather than a staircase.
    return (
        f"frame=none,xleftmargin=0pt,bgcolor={bg},"
        r"fontsize=\small,breaklines=true,breakanywhere=true,breakautoindent=false"
    )


_BOX_COLS = 71   # chars an output box holds at \small before a line overflows


def is_transcript(content: str) -> bool:
    """True if ``content`` carries a labelled show_turn transcript that already
    fits the output box, so it can render with breaklines off (see
    ``_inline_text_opts``) for pixel-exact pipe alignment.

    A transcript block is two or more lines sharing one ``| `` gutter column
    (<=15, with only a speaker label or spaces to its left) — the gutter width
    varies per cell, and a cell may open with a line of prose before the block.
    breaklines is disabled only when *every* line fits within ``_BOX_COLS``;
    a stale, too-wide transcript fails this guard and keeps wrapping until it is
    reflowed, so turning breaklines off can never push a line off the box."""
    lines = content.split("\n")
    cols = {}
    for ln in lines:
        if not ln.strip():
            continue
        j = ln.find("| ")
        if 0 <= j <= 15 and "|" not in ln[:j]:
            cols[j] = cols.get(j, 0) + 1
    if not cols or max(cols.values()) < 2:
        return False
    return max((len(ln) for ln in lines), default=0) <= _BOX_COLS


def is_python(content: str) -> bool:
    """Heuristic detection for Python source blocks."""

    sample = "\n".join(content.strip().splitlines()[:8])

    return bool(
        PYTHON_RE.search(sample)
        or ASSIGNMENT_RE.search(sample)
        or CALL_RE.search(sample)
    )


def _make_minted(lang: str, content: str, inline: bool,
                 cell_env: str = "") -> str:
    if inline:
        opts = (_INLINE_PYTHON_OPTS if lang == "python"
                else _inline_text_opts(cell_env, transcript=is_transcript(content)))
        return f"\\begin{{minted}}[{opts}]{{{lang}}}\n{content}\n\\end{{minted}}"
    return f"\\begin{{minted}}{{{lang}}}\n{content}\n\\end{{minted}}"


def replace_verbatim(tex: str) -> str:
    """
    Replace verbatim blocks with minted environments, wrapping tagged cells
    in semantic tcolorbox environments when magic comments are present.

    Untagged cells use the default minted style unchanged.
    """

    VERBATIM_RE = re.compile(
        r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}",
        re.DOTALL,
    )

    parts = []
    last_end = 0
    pending_output_env = [None]  # mutable list so closure can write to it

    for match in VERBATIM_RE.finditer(tex):
        gap = tex[last_end:match.start()]
        parts.append(gap)
        last_end = match.end()

        # A model's streaming output fragments sit flush against each other, so
        # an untagged verbatim right after a tagged code cell is genuinely that
        # cell's output.  But an *editorial* fenced block (an author-drawn
        # diagram in a markdown cell) is separated from the preceding output by
        # paragraph prose.  If real prose intervenes, this block cannot be a
        # continuation of the previous cell's output, so drop any pending output
        # env — otherwise the illustration is wrapped in the ochre MODEL OUTPUT
        # box and reads as if a model produced it.
        gap_prose = re.sub(r"\\[a-zA-Z]+|[{}]", "", gap)
        gap_is_editorial = bool(
            re.search(r"[A-Za-z]{3,}[^A-Za-z]+[A-Za-z]{3,}", gap_prose))
        if gap_is_editorial:
            pending_output_env[0] = None

        content = match.group(1)
        stripped = content.strip("\n")

        # MyST escapes Unicode math glyphs into LaTeX macros (→ becomes
        # \rightarrow, ↔ becomes \leftrightarrow, and so on) before we ever see
        # the verbatim block. Inside code and output these must be literal
        # characters, not math commands, or XeLaTeX prints the macro name itself
        # (e.g. a raw "\leftrightarrow" in a code cell). Restore the real glyphs.
        # Longest macros first so \leftrightarrow is handled before any shorter
        # macro could partially match it.
        for _macro, _glyph in sorted(
                UNICODE_MACRO_RESTORE.items(), key=lambda kv: -len(kv[0])):
            stripped = stripped.replace(_macro, _glyph)

        lines = stripped.splitlines()
        first = lines[0].strip() if lines else ""
        cell_env = None

        for marker, (code_env, out_env) in CELL_MARKERS.items():
            if first == marker:
                stripped = "\n".join(lines[1:]).strip("\n")
                cell_env = code_env
                pending_output_env[0] = out_env
                break

        lang = "python" if is_python(stripped) else "text"

        # Any untagged block after a tagged cell is an output — consume pending.
        # Since every code cell is tagged, untagged verbatim blocks are always
        # outputs (model text, formatted results, generated code, etc.).
        # We never clear pending on an untagged block — only a new tagged cell
        # (detected above via CELL_MARKERS) resets pending_output_env.
        if cell_env is None and pending_output_env[0]:
            cell_env = pending_output_env[0]
            # Keep pending for consecutive outputs from the same code cell.

        # Output environments always render as plain text — responses and
        # results are not source code, even if they happen to contain code.
        OUTPUT_ENVS = {"modeloutcell", "evaloutcell", "sysoutcell"}
        if cell_env in OUTPUT_ENVS:
            lang = "text"

        # An untagged text block introduced by prose is an author-drawn
        # illustration, not program output. Route it to the editorial example
        # box (teal rule, no "output" label) instead of the default output rule.
        if cell_env is None and gap_is_editorial and lang == "text":
            cell_env = "examplecell"

        minted = _make_minted(lang, stripped, inline=(cell_env is not None),
                              cell_env=cell_env or "")

        if cell_env:
            parts.append(f"\\begin{{{cell_env}}}\n{minted}\n\\end{{{cell_env}}}")
        else:
            parts.append(minted)

    parts.append(tex[last_end:])
    return "".join(parts)


def merge_consecutive_boxes(tex: str) -> str:
    """
    Merge runs of consecutive identical output tcolorboxes into one box.

    Streaming cells produce one notebook output entry per token, which
    expands into a separate verbatim → tcolorbox per token.  This pass
    collapses runs of adjacent same-environment boxes into a single box
    by stripping the interior \\end{ENV}…\\begin{ENV}…\\begin{minted} boundary.
    Applied iteratively until no further merges are possible.
    """
    for env in ("sysoutcell", "modeloutcell", "evaloutcell"):
        pat = re.compile(
            r"\\end\{minted\}\n"
            r"\\end\{" + re.escape(env) + r"\}\n"
            r"\n?"                                    # optional blank line between boxes
            r"\\begin\{" + re.escape(env) + r"\}\n"
            r"\\begin\{minted\}\[[^\]]*\]\{text\}\n",
        )
        prev = None
        while prev != tex:
            prev = tex
            tex = pat.sub("\n", tex)
    return tex


def inject_output_separators(tex: str) -> str:
    """
    Inject \\outputsep before every \\begin{minted}{text} block.

    Idempotent: running twice does not double-inject separators.
    The \\outputsep command is defined in the LaTeX template and renders
    a thin labelled rule between the white code cell and the dark terminal
    output block.
    """
    PLACEHOLDER = "%%OUTPUTSEP%%"
    # Protect instances that already have the separator
    tex = tex.replace(r"\outputsep" + "\n" + r"\begin{minted}{text}", PLACEHOLDER)
    # Inject before all remaining text minted blocks
    tex = tex.replace(r"\begin{minted}{text}", r"\outputsep" + "\n" + r"\begin{minted}{text}")
    # Restore protected instances
    tex = tex.replace(PLACEHOLDER, r"\outputsep" + "\n" + r"\begin{minted}{text}")
    return tex


# ---------------------------------------------------------------------------
# Back-of-book index injection
# ---------------------------------------------------------------------------

INDEX_TERMS_FILE = ROOT / "scripts" / "index_terms.txt"

# Front/back-matter pages never receive index entries.
INDEX_SKIP_SLUGS = {"citation", "toc", "cover", "front", "glossary"}

_VERBATIM_SPLIT_RE = re.compile(
    r"(\\begin\{verbatim\}.*?\\end\{verbatim\})",
    re.DOTALL,
)


def load_index_terms():
    """Parse scripts/index_terms.txt into (regex, key, slugs) triples.

    Each line: ``pattern :: Key [:: slugs=a,b,c] [:: cs]``. The pattern is a
    regex fragment matched at word boundaries, case-insensitive unless ``cs``.
    Missing file → no index entries (the build still succeeds).
    """
    terms = []
    if not INDEX_TERMS_FILE.exists():
        return terms
    for line in INDEX_TERMS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split("::")]
        pattern, key, slugs, flags = fields[0], fields[0], None, re.IGNORECASE
        for f in fields[1:]:
            if f.startswith("slugs="):
                slugs = set(f[len("slugs="):].split(","))
            elif f == "cs":
                flags = 0
            elif f:
                key = f
        terms.append((re.compile(rf"\b(?:{pattern})\b", flags), key, slugs))
    return terms


def _index_match_ok(line: str, m: re.Match) -> bool:
    """Reject matches where an \\index{} insertion could land inside LaTeX
    machinery: lines that are themselves commands/comments, matches directly
    inside a command argument brace, and matches inside inline math."""
    head = line.lstrip()
    if head.startswith("\\") or head.startswith("%"):
        return False
    # ":" and "/" guard label/URL interiors like \hyperref[chap:agentic] or
    # \href{https://example.com/agents}, where \index{} is fatal.
    if m.start() > 0 and line[m.start() - 1] in "{\\:/":
        return False
    hyper = line.rfind("\\hyperref[", 0, m.start())
    if hyper != -1 and line.find("]", hyper, m.start()) == -1:
        return False
    if line.count("$", 0, m.start()) % 2 == 1:
        return False
    return True


def inject_index_terms(tex: str, slug: str, terms) -> str:
    """Insert \\index{Key} after each term's first prose occurrence in this
    chapter. Operates outside verbatim blocks only, at most once per key per
    chapter, and skips keys already present (idempotent across reruns)."""
    if slug in INDEX_SKIP_SLUGS or not terms:
        return tex
    segments = _VERBATIM_SPLIT_RE.split(tex)
    for rx, key, slugs in terms:
        if slugs is not None and slug not in slugs:
            continue
        marker = f"\\index{{{key}}}"
        if any(marker in seg for seg in segments):
            continue
        placed = False
        for si in range(0, len(segments), 2):   # even indices = outside verbatim
            lines = segments[si].split("\n")
            for li, line in enumerate(lines):
                for m in rx.finditer(line):
                    if _index_match_ok(line, m):
                        lines[li] = line[:m.end()] + marker + line[m.end():]
                        placed = True
                        break
                if placed:
                    break
            if placed:
                segments[si] = "\n".join(lines)
                break
    return "".join(segments)


ALT_TEXTS_FILE = ROOT / "scripts" / "alt_texts.yml"


def load_alt_texts():
    """Parse scripts/alt_texts.yml into {slug: {ordinal: alt}}.

    Keyed by (chapter slug, 1-based figure ordinal in document order). That key
    is stable across rebuilds even when MyST re-hashes a regenerated chart's
    output filename, which a filename key would not be. Missing file or no
    PyYAML → no alt text (the build still succeeds). Only non-empty alt strings
    are returned, so figures still awaiting a description are simply skipped.
    """
    try:
        import yaml
    except ImportError:
        return {}
    if not ALT_TEXTS_FILE.exists():
        return {}
    raw = yaml.safe_load(ALT_TEXTS_FILE.read_text(encoding="utf-8")) or {}
    out = {}
    for slug, entries in raw.items():
        by_ord = {e["ordinal"]: (e.get("alt") or "").strip()
                  for e in (entries or []) if (e.get("alt") or "").strip()}
        if by_ord:
            out[slug] = by_ord
    return out


_LATEX_TEXT_ESCAPES = [
    ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
    ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
]


def _latex_escape_text(s: str) -> str:
    r"""Escape LaTeX specials so a plain-English alt string is safe inside
    \Description{...}. Most important is %, which would otherwise comment out
    the closing brace and cause a runaway-argument error even though
    \Description is a no-op in this article-class preview."""
    s = s.replace("\\", "\x00")          # stash backslashes, restore last
    for ch, rep in _LATEX_TEXT_ESCAPES:
        s = s.replace(ch, rep)
    return s.replace("\x00", r"\textbackslash{}")


_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics\b")


def inject_alt_texts(tex: str, slug: str, alt_by_slug) -> str:
    r"""Insert \Description{<alt>} after each \includegraphics, keyed by the
    image's 1-based ordinal within this chapter in document order (validated to
    match the notebook figure order by scripts/_alt_scaffold.py). Idempotent: a
    figure already followed by \Description is left untouched but still counts
    toward the ordinal; figures with no authored alt are skipped."""
    by_ord = alt_by_slug.get(slug)
    if not by_ord:
        return tex
    lines = tex.split("\n")
    out = []
    ordinal = 0
    for i, line in enumerate(lines):
        out.append(line)
        if _INCLUDEGRAPHICS_RE.search(line):
            ordinal += 1
            nxt = lines[i + 1].lstrip() if i + 1 < len(lines) else ""
            if ordinal in by_ord and not nxt.startswith(r"\Description"):
                indent = line[:len(line) - len(line.lstrip())]
                alt = _latex_escape_text(by_ord[ordinal])
                out.append(f"{indent}\\Description{{{alt}}}")
    return "\n".join(out)


def strip_escaped_html(tex: str) -> str:
    """Remove HTML markup that MyST leaks into LaTeX from {raw} html blocks.

    MyST converts HTML-only directive content to escaped LaTeX text rather than
    suppressing it.  In the TOC file this appears as a blob of CSS properties
    and \\textless div \\textgreater tags between the section heading and the
    actual \\vspace raw-latex block.  Strip everything in that gap.
    """
    # Remove the blob between \section{...} and the first \vspace that opens
    # the hand-written LaTeX TOC, keeping the section heading and everything after.
    tex = re.sub(
        r"(\\section\{[^}]*\})\n.*?(?=\\vspace)",
        r"\1\n\n",
        tex,
        count=1,
        flags=re.DOTALL,
    )
    # The preface poem centers itself with paired {raw} fences (latex + html).
    # MyST leaks the html pair into the tex as escaped tag lines; drop exactly
    # those, leaving the \begin{center}...\end{center} pair from the latex side.
    tex = re.sub(r"\\textless\s*div style=\"text-align:center[^\n]*\n?", "", tex)
    tex = re.sub(r"\\textless\s*/div\\textgreater\s*\n?", "", tex)
    return tex


def inject_chapter_label(tex: str, slug: str) -> str:
    """Add \\label{chap:<slug>} after the first \\section{...} or \\section*{...} if not present."""
    label = f"\\label{{chap:{slug}}}"
    if label in tex:
        return tex
    return re.sub(
        r"(\\section\*?\{[^}]*\})",
        lambda m: m.group(0) + "\n" + label,
        tex,
        count=1,
    )


CHAPTER_SLUGS = {
    "front", "intro", "prompting", "tokens", "semantics",
    "metacoding", "augmentation", "agentic", "thinking", "mm", "design",
    "responsible", "efficiency", "glossary",
}


def convert_chapter_hrefs(tex: str) -> str:
    r"""Turn myst's URL form of a cross-chapter reference into an internal link.

    A markdown reference to another chapter's ``(chap:<slug>)=`` target
    exports as \href{/<slug>}{text} (the HTML site URL, dead in a PDF).
    Rewrite it to \hyperref[chap:<slug>]{text}, pointing at the label that
    inject_chapter_label guarantees every chapter carries.
    """
    return re.sub(
        r"\\href\{/(" + "|".join(sorted(CHAPTER_SLUGS)) + r")\}",
        r"\\hyperref[chap:\1]",
        tex,
    )


UNNUMBERED_SLUGS = {"citation", "toc", "glossary", "front"}


def make_section_unnumbered(tex: str, slug: str) -> str:
    r"""Star every sectioning command on pages that carry no chapter number.

    Applied to the How-to-Cite page, the hand-built TOC, the glossary, and
    the Preface (front.ipynb), so the section counter is untouched and the
    Introduction opens as chapter 1. All heading levels are starred because
    a \subsection under a starred \section would number itself 0.1. Starred
    headings produce no PDF outline entry, so one bookmark is added at the
    page's first heading. The Preface and Glossary additionally get an
    unnumbered chapter-level line in the Detailed Contents (the \phantomsection
    anchors the entry's hyperlink at the heading; a starred \section sets no
    target of its own), matching their rows in the hand-built TOC."""
    if slug not in UNNUMBERED_SLUGS:
        return tex
    tex = re.sub(r"\\((?:sub)*section)\{", r"\\\1*{", tex)
    m = re.search(r"\\section\*\{([^}]*)\}", tex)
    if m and r"\pdfbookmark" not in tex:
        tex = (tex[:m.start()]
               + f"\\pdfbookmark[1]{{{m.group(1).strip()}}}{{bm-{slug}}}\n"
               + tex[m.start():])
    if slug in {"front", "glossary"} and r"\addcontentsline" not in tex:
        m = re.search(r"\\section\*\{([^}]*)\}", tex)
        if m:
            tex = (tex[:m.end()]
                   + "\n\\phantomsection\\addcontentsline{toc}{section}"
                   + f"{{{m.group(1).strip()}}}"
                   + tex[m.end():])
    return tex


_EPIGRAPH_RE = re.compile(
    r'\\textit\{([^{}]+)\}\n\n---\s+(.+)',
)


def wrap_epigraphs(tex: str) -> str:
    r"""Wrap chapter epigraphs in \chapterepigraph{quote}{attribution}."""
    return _EPIGRAPH_RE.sub(
        lambda m: f'\\chapterepigraph{{{m.group(1)}}}{{{m.group(2)}}}',
        tex,
    )


_VERSE_RE = re.compile(
    r"(?<!\\begin\{center\}\n)"
    r"((?:\\textit\{[^{}\n]*\}\\newline\n){3,}---[^\n]*\n)"
)


def center_verse(tex: str) -> str:
    r"""Center multi-line verse: a run of 3+ hard-broken italic lines ending
    in an --- attribution (Sophia's preface poem). The notebook stays plain
    markdown; mystmd's {raw} latex directive can't express this without
    "Unhandled TEX conversion" errors. The lookbehind keeps a second pass
    from nesting center environments."""
    return _VERSE_RE.sub(
        lambda m: "\\begin{center}\n" + m.group(1) + "\\end{center}\n",
        tex,
    )


_GENAI_STORY_RE = re.compile(
    r"\\begin\{framed\}\s*"
    r"\\textbf\{GenAI Story:\s*(?P<title>.+?)\}\\\\\s*"
    r"(?P<body>.*?)"
    r"\\end\{framed\}",
    re.DOTALL,
)


def route_genai_stories(tex: str) -> str:
    """Swap framed→genaistorybox for admonitions whose title starts with
    'GenAI Story:'. The 'GenAI Story:' prefix becomes the box's coloured
    title bar; the specific story title stays bolded in the body."""
    def _swap(m: re.Match) -> str:
        title = m.group("title").strip()
        body = m.group("body")
        return (
            "\\begin{genaistorybox}\n"
            f"\\textbf{{{title}}}\\\\\n"
            f"{body}"
            "\\end{genaistorybox}"
        )
    return _GENAI_STORY_RE.sub(_swap, tex)


_FAQ_TIP_RE = re.compile(
    r"(\\begin\{framed\}\s*\\textbf\{Q: [^\n]*?\})\\\\"
)


def underline_faq_titles(tex: str) -> str:
    r"""Swap the line break after a tip admonition's bold 'Q: ...' header for
    \faqtitlerule (template.tex), a thin teal rule under the question. Keyed
    on the 'Q: ' prefix so notes and other tips are untouched. Idempotent:
    once the \\ is consumed the pattern no longer matches."""
    return _FAQ_TIP_RE.sub(r"\1\\faqtitlerule", tex)


def enlarge_chart_figures(tex: str) -> str:
    r"""Rewrite MyST's default chart width so figures fill more of the page.

    MyST renders every matplotlib cell-output figure at
    `\includegraphics[width=0.7\linewidth]{...}`. We bump that one default up to
    FIGURE_WIDTH. Idempotent and surgical: only the exact MyST default is
    matched, so photographs and diagrams carrying their own explicit widths keep
    them.
    """
    return tex.replace(
        f"width={MYST_DEFAULT_FIGURE_WIDTH}\\linewidth",
        f"width={FIGURE_WIDTH}\\linewidth",
    )


_IMG_LINE_RE = re.compile(r"^\\includegraphics\[[^\]]*\]\{[^{}]*\}$")


def _full_textit(par: str):
    r"""Return the inner text if ``par`` is a single whole \textit{...}
    paragraph (the book's caption convention for generated images), else
    None. Brace-aware so captions containing \cite{...} or \href{...}{...}
    survive; a paragraph that merely *starts* italic fails the
    closes-at-the-end check and stays body prose."""
    body = par.rstrip()
    if not body.startswith(r"\textit{") or not body.endswith("}"):
        return None
    i, depth, n = 0, 0, len(body)
    while i < n:
        ch = body[i]
        if ch == "\\" and i + 1 < n and body[i + 1] in "{}":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return body[len(r"\textit{"):i] if i == n - 1 else None
        i += 1
    return None


def wrap_figures(tex: str, numbered: bool = True) -> str:
    r"""Wrap each standalone cell-output image in a numbered figure block.

    ``numbered=False`` (the Preface) keeps the caption text but drops the
    counter: \captionof* for captioned images, no label line for bare ones,
    and nothing is written to the List of Figures.

    MyST emits every matplotlib/cell-output image as a bare \includegraphics
    paragraph: no number, no caption, no way for prose to cite it. We wrap
    each one in the template's non-floating ``figurebox`` so it keeps its
    exact position in the reading flow but increments the per-chapter figure
    counter:

      * next paragraph entirely \textit{...}  → absorbed as \captionof caption
      * anything else                         → \figlabel ("Figure 4.2" only;
                                                 the explanation stays in prose)

    Images already inside a figure environment (MyST {figure} directives in
    the multimodal chapter) are preceded by \centering, not a blank line, so
    they are skipped here — ``unstar_captions`` numbers those instead. The
    same guard makes this pass idempotent.
    """
    lines = tex.split("\n")
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        prev_blank = not out or out[-1].strip() == ""
        next_blank = i + 1 >= n or lines[i + 1].strip() == ""
        if not (_IMG_LINE_RE.match(line) and prev_blank and next_blank):
            out.append(line)
            i += 1
            continue
        j = i + 1
        while j < n and lines[j].strip() == "":
            j += 1
        k = j
        while k < n and lines[k].strip() != "":
            k += 1
        caption = _full_textit("\n".join(lines[j:k])) if j < n else None
        star = "" if numbered else "*"
        out.append(r"\begin{figurebox}")
        out.append(line)
        if caption is not None:
            out.append(rf"\captionof{star}{{figure}}{{{caption}}}")
            out.append(r"\end{figurebox}")
            i = k          # caption paragraph consumed into \captionof
        else:
            if numbered:
                out.append(r"\figlabel")
            out.append(r"\end{figurebox}")
            i += 1         # following blank line and prose stay untouched
    return "\n".join(out)


# Markdown tables can't express column widths, so MyST gives every column an
# equal fraction of the line. That breaks when a column holds an unbreakable
# \texttt identifier wider than its share (nomic-embed-text overran Params by
# 24pt in the front-matter lineup). Override fractions here, keyed by the
# table's exact header row; fractions must sum to 1.0 and match the column
# count or the override is skipped with a warning.
TABLE_COL_WIDTHS = {
    r"Model & Params & Size & Best at & Downloads \\": (0.27, 0.11, 0.12, 0.35, 0.15),
}


def tune_table_widths(tex: str) -> str:
    r"""Rewrite MyST's equal p{\dimexpr ...} column spec for known tables."""
    lines = tex.split("\n")
    for header, widths in TABLE_COL_WIDTHS.items():
        for i, line in enumerate(lines):
            if line.strip() != header:
                continue
            for j in range(max(0, i - 3), i):
                if not lines[j].startswith(r"\begin{tabular}{p{\dimexpr"):
                    continue
                if lines[j].count(r"p{\dimexpr") != len(widths):
                    print(f"  WARNING: width override for {header!r} has "
                          f"{len(widths)} fractions but the table does not; skipped")
                    continue
                spec = "".join(
                    rf"p{{\dimexpr {w:.3f}\linewidth-2\tabcolsep}}"
                    for w in widths)
                lines[j] = rf"\begin{{tabular}}{{{spec}}}"
    return "\n".join(lines)


def wrap_tables(tex: str, numbered: bool = True) -> str:
    r"""Wrap each standalone markdown-derived table in a numbered table block.

    MyST emits a markdown table as ``\bigskip\noindent`` + a
    ``\begin{tabular}{p{\dimexpr ...}}`` block. We swap the bigskip for the
    template's ``tablebox`` and number the table, absorbing a following
    fully-italic paragraph as the caption (placed above the tabular, where
    table captions conventionally sit). Tables inside framed admonitions
    keep their box title and are left alone, as is anything whose column
    spec doesn't carry MyST's ``p{\dimexpr`` fingerprint (e.g. the
    hand-built TOC). Idempotent: wrapping removes the bigskip line the
    pattern keys on.
    """
    lines = tex.split("\n")
    out = []
    i, n = 0, len(lines)
    depth = 0
    while i < n:
        line = lines[i]
        if re.match(r"\\begin\{(framed|genaistorybox)\}", line):
            depth += 1
        elif re.match(r"\\end\{(framed|genaistorybox)\}", line):
            depth -= 1
        if not (line == r"\bigskip\noindent" and depth == 0
                and i + 1 < n
                and lines[i + 1].startswith(r"\begin{tabular}{p{\dimexpr")):
            out.append(line)
            i += 1
            continue
        j = i + 1
        while j < n and lines[j].strip() != r"\end{tabular}":
            j += 1
        body = lines[i + 1:j + 1]          # \begin{tabular} ... \end{tabular}
        # MyST closes every table with a lone \bigskip paragraph; look past it
        # for the caption (and consume it on absorb — the box has its own skip).
        k = j + 1
        while k < n and lines[k].strip() == "":
            k += 1
        if k < n and lines[k].strip() == r"\bigskip" and (
                k + 1 >= n or lines[k + 1].strip() == ""):
            k += 1
            while k < n and lines[k].strip() == "":
                k += 1
        m = k
        while m < n and lines[m].strip() != "":
            m += 1
        caption = _full_textit("\n".join(lines[k:m])) if k < n else None
        star = "" if numbered else "*"
        out.append(r"\begin{tablebox}")
        if caption is not None:
            out.append(rf"\captionof{star}{{table}}{{{caption}}}")
        elif numbered:
            out.append(r"\tablabel")
        out.extend(body)
        out.append(r"\end{tablebox}")
        i = m if caption is not None else j + 1
    return "\n".join(out)


def unstar_captions(tex: str) -> str:
    r"""\caption*{...} → \caption{...} inside MyST figure environments.

    Without the caption package \caption* misparses — the star leaks into
    the output as a literal "Figure 1: *" line. With the package loaded the
    starred form would merely stay unnumbered. Either way the book wants
    these figures numbered like every other, so drop the star."""
    return tex.replace(r"\caption*{", r"\caption{")


def fix_math_spacing(tex: str) -> str:
    """Fix MyST stripping spaces before inline math mode.

    MyST converts markdown `word $math$ word` → LaTeX `word$math$ word`,
    causing "word$" to render with no space. Restore the space.
    Regex: word char (letter/digit/}) followed by $ → add space between them.
    """
    return re.sub(r'(\w|\})\$', r'\1 $', tex)


# Slugs that never carry a chapter reference list: structural pages and front/
# back matter. Every other page gets its own biblatex refsection so the works it
# cites print at its end (Springer house style) rather than in one list at the
# back of the book.
_NO_BIB_SLUGS = {"toc", "citation", "frontmatter", "glossary", "cover"}


def wrap_chapter_bibliography(tex: str, slug: str) -> str:
    r"""Wrap a chapter in a biblatex refsection and print its references at the end.

    Each chapter cites a different subset of references.bib; biblatex's refsection
    environment scopes the following \printbibliography to just the works cited
    inside it, so the chapter closes with its own numbered, citation-ordered
    "References" list. Idempotent, and a no-op for pages that cite nothing."""
    if slug in _NO_BIB_SLUGS:
        return tex
    if r"\begin{refsection}" in tex:        # already wrapped on a previous run
        return tex
    if r"\cite" not in tex:                  # nothing to reference
        return tex
    return (
        "\\begin{refsection}\n"
        + tex.rstrip("\n")
        + "\n\n\\printbibliography[heading=subbibliography]\n"
        "\\end{refsection}\n"
    )


def post_process_file(path: Path, index_terms=None, alt_texts=None):
    """Apply post-processing to a single tex file."""

    original = path.read_text(encoding="utf-8")
    updated = strip_escaped_html(original)
    # Derive slug from filename: programming-genai-prompting.tex -> prompting
    slug = path.stem.replace("programming-genai-", "")
    updated = make_section_unnumbered(updated, slug)   # \section{} → \section*{} for unnumbered pages
    updated = inject_chapter_label(updated, slug)
    updated = convert_chapter_hrefs(updated)
    updated = wrap_epigraphs(updated)
    updated = center_verse(updated)
    updated = route_genai_stories(updated)
    updated = underline_faq_titles(updated)
    updated = enlarge_chart_figures(updated)
    numbered = slug not in UNNUMBERED_SLUGS
    updated = wrap_figures(updated, numbered)
    updated = inject_alt_texts(updated, slug, alt_texts or {})
    bare = updated.count(r"\figlabel")
    if bare:
        print(f"  WARNING: {slug}: {bare} figure(s) without an italic caption "
              "paragraph — numbered but left out of the List of Figures")
    if slug not in ("toc", "citation", "frontmatter", "glossary"):
        updated = tune_table_widths(updated)
        updated = wrap_tables(updated, numbered)
    updated = unstar_captions(updated)
    updated = fix_math_spacing(updated)
    updated = inject_index_terms(updated, slug, index_terms or [])
    updated = replace_verbatim(updated)
    updated = merge_consecutive_boxes(updated)
    updated = inject_output_separators(updated)
    updated = wrap_chapter_bibliography(updated, slug)

    if updated != original:
        path.write_text(updated, encoding="utf-8")
        print(f"  processed: {path.name}")


def inject_lists():
    r"""Add the Detailed Contents and the Lists after the hand-built TOC page.

    The hand-built TOC (toc.md) stays the chapters-at-a-glance opener; the
    native \tableofcontents that follows it is the deep version, capped at
    tocdepth 2 so every chapter lists its H2 sections but H3s stay out.
    Preface/Glossary rows come from make_section_unnumbered, the Index row
    from imakeidx's intoc option. \contentsname is renamed inside a group so
    the page heading reads "Detailed Contents".

    The main tex is regenerated by every myst build, so this runs each time;
    the guard makes a manual second invocation harmless. Every figure and
    table is captioned via wrap_figures/wrap_tables, so the lists are
    complete rather than rows of bare numbers."""
    main_tex = EXPORTS / "programming-genai.tex"
    if not main_tex.exists():
        return
    tex = main_tex.read_text(encoding="utf-8")
    if r"\listoffigures" in tex:
        return
    anchor = "\\include{programming-genai-toc}"
    if anchor not in tex:
        print("  WARNING: toc include not found; skipping Detailed Contents")
        return
    tex = tex.replace(
        anchor,
        anchor
        + "\n\\clearpage"
        + "\n\\pdfbookmark[1]{Detailed Contents}{bm-dtoc}"
        + "\n\\begingroup"
        + "\n\\renewcommand{\\contentsname}{Detailed Contents}"
        + "\n\\setcounter{tocdepth}{2}"
        + "\n\\tableofcontents"
        + "\n\\endgroup"
        + "\n\\clearpage\n\\listoffigures\n\\listoftables\n\\clearpage",
        1,
    )
    main_tex.write_text(tex, encoding="utf-8")
    print("  injected Detailed Contents + List of Figures / List of Tables")


def inject_front_matter_pagination():
    r"""Roman page numbers up to the Preface, arabic from the Introduction.

    The cover, title, TOC, lists, and Preface paginate i, ii, iii…;
    \pagenumbering{arabic} resets to page 1 where the Introduction opens.
    Page anchors are off during the roman stretch so hyperref never sees
    two pages with the same name. Runs on the regenerated main tex each
    build; the guard makes a second manual invocation harmless."""
    main_tex = EXPORTS / "programming-genai.tex"
    if not main_tex.exists():
        return
    tex = main_tex.read_text(encoding="utf-8")
    if r"\pagenumbering{roman}" in tex:
        return
    anchor = "\\include{programming-genai-intro}"
    if "\\begin{document}" not in tex or anchor not in tex:
        print("  WARNING: pagination anchors not found; skipping roman front matter")
        return
    tex = tex.replace(
        "\\begin{document}",
        "\\begin{document}\n\\pagenumbering{roman}\\hypersetup{pageanchor=false}",
        1,
    )
    tex = tex.replace(
        anchor,
        "\\clearpage\\pagenumbering{arabic}\\hypersetup{pageanchor=true}\n" + anchor,
        1,
    )
    main_tex.write_text(tex, encoding="utf-8")
    print("  front matter paginated in roman numerals")


def strip_injected_natbib():
    r"""Remove the \usepackage{natbib} MyST injects into the main tex.

    MyST renders citations with natbib commands (\citet) and auto-adds
    \usepackage{natbib} to the generated main tex, separate from the template.
    We drive references through biblatex instead (per-chapter refsections), and
    biblatex aborts with "Incompatible package 'natbib'" if natbib is also
    loaded — recording zero citations and emptying every reference list. The
    biblatex natbib=true compat layer already provides \citet/\cite, so the real
    natbib package is redundant here and must go. Idempotent."""
    main_tex = EXPORTS / "programming-genai.tex"
    if not main_tex.exists():
        return
    tex = main_tex.read_text(encoding="utf-8")
    new = re.sub(r"^\\usepackage(?:\[[^\]]*\])?\{natbib\}\n", "", tex, flags=re.M)
    if new != tex:
        main_tex.write_text(new, encoding="utf-8")
        print("  stripped MyST-injected \\usepackage{natbib} (biblatex drives refs)")


def clean_main_bib():
    """Drop MyST-injected duplicate `howpublished` URLs from the generated bib.

    MyST copies each `url` field into a `howpublished` field for @misc entries,
    which makes the .bst print the URL twice in the reference list. Our source
    references.bib never sets howpublished, so every one here is a duplicate and
    is safe to remove.
    """
    bib = EXPORTS / "main.bib"
    if not bib.exists():
        return
    lines = bib.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [ln for ln in lines if not re.match(r"\s*howpublished\s*=", ln)]
    removed = len(lines) - len(kept)
    if removed:
        bib.write_text("".join(kept), encoding="utf-8")
        print(f"  removed {removed} duplicate howpublished URL(s) from main.bib")


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def build_tex(env):
    """Steps 1–2: MyST → TeX, then post-process every chapter and the main tex.

    Leaves exports/programming-genai.tex and each programming-genai-<slug>.tex
    ready for latexmk. Shared by both the whole-book and per-chapter builds."""

    print("\n── Step 1: myst build --tex ───────────────────────────────")

    # mystmd's tex export mangles hyphens inside verbatim blocks ("600-page"
    # -> "600 -page"); patch the installed bundle first (idempotent, see
    # scripts/patch_mystmd.py).
    from patch_mystmd import patch as patch_mystmd
    patch_mystmd()

    run(["uv", "run", "myst", "build", "--tex"], env=env)

    print("\n── Step 2: post-processing verbatim blocks ───────────────")

    chapter_files = sorted(EXPORTS.glob("programming-genai-*.tex"))

    index_terms = load_index_terms()
    alt_texts = load_alt_texts()

    for tex_file in chapter_files:
        post_process_file(tex_file, index_terms, alt_texts)

    n_alt = sum(len(v) for v in alt_texts.values())
    print(f"  done ({len(chapter_files)} chapter files, "
          f"{len(index_terms)} index terms, {n_alt} figure alt texts)")

    inject_lists()
    inject_front_matter_pagination()
    strip_injected_natbib()
    clean_main_bib()


def latexmk(stem, env):
    """Run the book's latexmk recipe on ``<stem>.tex`` inside exports/."""
    run(
        [
            "latexmk",
            "-xelatex",
            # No -bibtex: biblatex's per-chapter reference lists use the biber
            # backend, which latexmk auto-detects and runs from the .bcf file.
            "-shell-escape",
            "-interaction=batchmode",
            # TOC + LoF/LoT + index + bibliography legitimately need more than
            # latexmk's default 5 passes to converge; below the limit it aborts
            # with "xelatex needed too many passes" and no fresh PDF.
            "-e", "$max_repeat=9",
            "-f",
            f"{stem}.tex",
        ],
        cwd=str(EXPORTS),
        env=env,
    )


def _pdf_is_fresh(pdf: Path, src: Path) -> bool:
    """True if ``pdf`` was (re)built from ``src`` on this run.

    latexmk exits 12 both for benign warnings and for real failures like
    "needed too many passes", so a returncode check alone can't tell a good
    build from one that left last run's PDF behind. The tex is rewritten every
    build, so a PDF older than its source was not produced by this run."""
    return pdf.exists() and pdf.stat().st_mtime >= src.stat().st_mtime


def compile_book(env):
    """Step 3: compile the whole-manuscript PDF."""

    print("\n── Step 3: latexmk (whole manuscript) ────────────────────")

    latexmk("programming-genai", env)

    pdf = EXPORTS / "programming-genai.pdf"
    main_tex = EXPORTS / "programming-genai.tex"

    if _pdf_is_fresh(pdf, main_tex):
        print(f"\n✓ PDF built successfully:\n"
              f"  {pdf}\n"
              f"  ({pdf.stat().st_size // 1024} KB)")
    else:
        print("\n✗ PDF stale or missing — latexmk did not finish.\n"
              "Check:\n"
              "  exports/programming-genai.log")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Per-chapter PDFs (Springer submission)
# ---------------------------------------------------------------------------

# Springer asks for one PDF per chapter with all fonts embedded, named with the
# first author's last name and the chapter number (their example: "Myers-Chap1").
# Front/back matter that carries no chapter number is named by its title.
AUTHOR_LAST = "Abuah"

# (slug, chapter number or None, display title) in reading order. The number
# both names the file and pre-sets \section, so a chapter compiled on its own
# still shows its real chapter and figure numbers (Prompting = 2, Figure 2.1, …)
# because the template numbers figures within \section.
CHAPTER_SEQUENCE = [
    ("front",        None, "Preface"),
    ("intro",        1,    "Introduction"),
    ("prompting",    2,    "Prompting"),
    ("tokens",       3,    "Tokens"),
    ("semantics",    4,    "Semantics"),
    ("metacoding",   5,    "Metacoding"),
    ("augmentation", 6,    "Augmentation"),
    ("agentic",      7,    "Agentic"),
    ("thinking",     8,    "Thinking"),
    ("mm",           9,    "Multimodal"),
    ("efficiency",   10,   "Efficiency"),
    ("responsible",  11,   "Responsible"),
    ("glossary",     None, "Glossary"),
]

CHAPTER_PDF_DIR = EXPORTS / "chapters"


def _chapter_pdf_stem(number, title) -> str:
    """Springer submission filename for a chapter PDF (no extension)."""
    if number is None:
        return f"{AUTHOR_LAST}-{title}"
    return f"{AUTHOR_LAST}-Chap{number}"


def _split_preamble() -> str:
    r"""Return the preamble of the generated main tex (everything before
    \begin{document}). Read after build_tex() so MyST's injected natbib is
    already stripped and biblatex drives references cleanly."""
    main_tex = EXPORTS / "programming-genai.tex"
    tex = main_tex.read_text(encoding="utf-8")
    idx = tex.find("\\begin{document}")
    if idx == -1:
        print("ERROR: \\begin{document} not found in main tex", file=sys.stderr)
        sys.exit(1)
    return tex[:idx]


def _clean_chapter_artifacts(stem):
    """Remove a chapter build's scratch files (everything but its PDF)."""
    for path in EXPORTS.glob(f"{stem}.*"):
        if path.suffix != ".pdf":
            path.unlink()
    minted = EXPORTS / f"_minted-{stem}"
    if minted.is_dir():
        shutil.rmtree(minted)


def build_chapter_pdfs(env, only=None):
    r"""Build one standalone, font-embedded PDF per chapter for Springer.

    Each chapter is wrapped in the main document's preamble and compiled on its
    own with cwd=exports (so ../fonts/ and files/ resolve). \section is pre-set
    so chapter and figure numbers match the full book, and every chapter already
    carries its own biblatex refsection, so its reference list prints at its end.
    Cross-chapter links are plain \hyperref text and render fine; there are no
    \ref/\pageref across chapters to come out undefined. PDFs land in
    exports/chapters/; scratch files are cleaned unless a chapter fails."""

    print("\n── Per-chapter PDFs (Springer submission) ────────────────")

    preamble = _split_preamble()
    CHAPTER_PDF_DIR.mkdir(exist_ok=True)

    targets = [c for c in CHAPTER_SEQUENCE if only is None or c[0] in only]
    if not targets:
        print(f"  no chapters match {sorted(only)}")
        sys.exit(1)

    built, failed = [], []
    for slug, number, title in targets:
        chapter_tex = EXPORTS / f"programming-genai-{slug}.tex"
        if not chapter_tex.exists():
            print(f"  SKIP {slug}: {chapter_tex.name} not found")
            continue

        stem = _chapter_pdf_stem(number, title)
        setcounter = (f"\\setcounter{{section}}{{{number - 1}}}\n"
                      if number is not None else "")
        wrapper = (
            preamble
            + "\\begin{document}\n"
            + "\\pagenumbering{arabic}\n"
            + setcounter
            + f"\\input{{programming-genai-{slug}}}\n"
            + "\\end{document}\n"
        )
        wrapper_path = EXPORTS / f"{stem}.tex"
        wrapper_path.write_text(wrapper, encoding="utf-8")

        print(f"\n  → {stem}.pdf  ({title})")
        latexmk(stem, env)

        out_pdf = EXPORTS / f"{stem}.pdf"
        if _pdf_is_fresh(out_pdf, wrapper_path):
            dest = CHAPTER_PDF_DIR / out_pdf.name
            shutil.move(str(out_pdf), str(dest))
            _clean_chapter_artifacts(stem)
            built.append(dest)
        else:
            failed.append(stem)
            print(f"    ✗ {stem} failed — leaving artifacts in exports/ "
                  f"(see {stem}.log)")

    print(f"\n  {len(built)} chapter PDF(s) → {CHAPTER_PDF_DIR}")
    for dest in built:
        print(f"    {dest.name}  ({dest.stat().st_size // 1024} KB)")
    if failed:
        print(f"  ✗ {len(failed)} failed: {', '.join(failed)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Build the Programming GenAI PDF(s).")
    parser.add_argument(
        "--chapters", action="store_true",
        help="build one standalone PDF per chapter (Springer submission) "
             "instead of the whole-book PDF")
    parser.add_argument(
        "--all", action="store_true",
        help="build the whole-book PDF and the per-chapter PDFs")
    parser.add_argument(
        "--only", metavar="SLUG", action="append",
        help="restrict per-chapter builds to these slug(s); repeatable "
             "(e.g. --only prompting --only tokens). Implies --chapters")
    args = parser.parse_args()

    os.chdir(ROOT)

    # Ensure TeX binaries are visible on macOS
    env = os.environ.copy()
    env["PATH"] = f"/Library/TeX/texbin:{env['PATH']}"

    build_tex(env)

    do_chapters = bool(args.chapters or args.all or args.only)
    do_book = bool(args.all) or not do_chapters

    if do_book:
        compile_book(env)
    if do_chapters:
        build_chapter_pdfs(env, only=set(args.only) if args.only else None)


if __name__ == "__main__":
    main()