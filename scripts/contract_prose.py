"""
contract_prose.py
Convert formal negations to contractions (cannot -> can't, is not -> isn't)
in book prose: markdown cells of chapters/*.ipynb plus chapters/glossary.md.

Protected regions are never touched:
  - code cells and all cell outputs (frozen experiment data)
  - fenced code blocks and non-prose directives (epigraph, figure, table, ...)
  - inline code spans
  - double-quoted text, straight or curly (captured model/prompt wording)
  - markdown blockquote lines and HTML comments
Emphasis like "is *not*" never matches the patterns, so deliberate stress
survives on its own. "must not" is excluded (the Feynman principle stays
verbatim) and lowercase "let us" is excluded ("embeddings let us play" has
"let" as a main verb); only sentence-initial "Let us" becomes "Let's".

Usage: python scripts/contract_prose.py --dry-run [files...]
       python scripts/contract_prose.py [files...]
"""
import json, re, sys, glob
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_FILES = sorted(glob.glob(str(ROOT / "chapters" / "*.ipynb"))) + [
    str(ROOT / "chapters" / "glossary.md")
]

# First-letter case is preserved by the capture group; apostrophes are
# straight to match the book's existing contractions.
RULES = [
    (re.compile(r"\b([Cc])annot\b(?!-)"), r"\1an't"),
    (re.compile(r"\b([Ii])s not\b(?!-)"), r"\1sn't"),
    (re.compile(r"\b([Aa])re not\b(?!-)"), r"\1ren't"),
    (re.compile(r"\b([Ww])as not\b(?!-)"), r"\1asn't"),
    (re.compile(r"\b([Ww])ere not\b(?!-)"), r"\1eren't"),
    (re.compile(r"\b([Dd])o not\b(?!-)"), r"\1on't"),
    (re.compile(r"\b([Dd])oes not\b(?!-)"), r"\1oesn't"),
    (re.compile(r"\b([Dd])id not\b(?!-)"), r"\1idn't"),
    (re.compile(r"\b([Hh])as not\b(?!-)"), r"\1asn't"),
    (re.compile(r"\b([Hh])ave not\b(?!-)"), r"\1aven't"),
    (re.compile(r"\b([Hh])ad not\b(?!-)"), r"\1adn't"),
    (re.compile(r"\bwill not\b(?!-)"), "won't"),
    (re.compile(r"\bWill not\b(?!-)"), "Won't"),
    (re.compile(r"\b([Ww])ould not\b(?!-)"), r"\1ouldn't"),
    (re.compile(r"\b([Ss])hould not\b(?!-)"), r"\1houldn't"),
    (re.compile(r"\b([Cc])ould not\b(?!-)"), r"\1ouldn't"),
    (re.compile(r"\bLet us\b"), "Let's"),
]

# Positive contractions need a following word: a clause-final pair ("as
# simple as it is", "tell me what it is.") and the i.e. sense of "that is,"
# must stay uncontracted. The lookahead enforces space(s) then prose.
_NEXT = r"(?= +[\w*_`\"“'(\[])"

# "it" and "that" are the only pronouns here whose subject and object forms
# match, so "of it is short" / "under it is the page" can fool the rules:
# the pronoun belongs to the preposition and "is" to an earlier subject.
_PREPS = ("of", "to", "under", "over", "behind", "beneath", "above", "below",
          "around", "inside", "outside", "near", "beyond", "against", "from",
          "with", "without", "about", "on", "in", "at", "for", "by",
          "between", "through", "across", "onto", "upon", "within",
          "underneath", "beside", "besides", "atop", "amid", "alongside",
          "toward", "towards")
# "before"/"after" are deliberately absent: in this book they precede clauses
# ("before it is embedded"), where contraction is correct.


def _prep_before(m):
    before = m.string[max(0, m.start() - 12):m.start()]
    w = re.search(r"([A-Za-z]+) $", before)
    return bool(w) and w.group(1).lower() in _PREPS


def _it_is(m):
    """Reject 'it is' when 'it' is the object of a preceding infinitive
    ("the way to answer it is to run..."), gerund ("evaluating it is like
    cooking"), or preposition ("the spine of it is short"), and when 'is'
    carries an elided predicate after 'as' ("as it is on Well")."""
    before = m.string[max(0, m.start() - 30):m.start()]
    after = m.string[m.end():m.end() + 10]
    if re.search(r"\bto [a-z]+ $", before) or re.search(r"\b[a-z]+ing $", before):
        return None
    if _prep_before(m):
        return None
    if before.endswith("as ") and re.match(r" +(on|in|at|with|for|of|by)\b", after):
        return None
    return m.group(1) + "t's"


def _no_prep(repl):
    def f(m):
        return None if _prep_before(m) else m.expand(repl)
    return f


def _here_is(m):
    """Contract only presentational 'here is the catch', never 'the
    classifier here is gemma4', where 'here' modifies the noun before it.
    Presentational uses sit clause-initial: at the start of a line, or
    preceded by punctuation or a light connective, not by a content word."""
    line = m.string[:m.start()].rsplit("\n", 1)[-1].rstrip()
    w = re.search(r"([A-Za-z]+)$", line)
    if w and w.group(1).lower() not in ("so", "and", "but", "now", "then",
                                        "or", "yet"):
        return None
    return m.group(1) + "ere's"


# One-off sites no structural guard can catch (locative "there", etc.).
# Matched against the text around each candidate; survives re-runs.
NEVER = (
    "The tradeoff there is reliability",
    "somewhere out there is a phrasing",
    # Feynman's wording, woven into prose with a cite; keep it verbatim
    "you are the easiest person to fool",
)


RULES += [
    (re.compile(r"\b([Ii])t is\b" + _NEXT), _it_is),
    (re.compile(r"\b([Tt])hat is\b" + _NEXT), _no_prep(r"\1hat's")),
    (re.compile(r"\b([Tt])here is\b" + _NEXT), r"\1here's"),
    (re.compile(r"\b([Hh])ere is\b" + _NEXT), _here_is),
    (re.compile(r"\b([Ww])hat is\b" + _NEXT), r"\1hat's"),
    (re.compile(r"\bI am\b" + _NEXT), "I'm"),
    (re.compile(r"\b([Ww])e are\b" + _NEXT), r"\1e're"),
    (re.compile(r"\b([Yy])ou are\b" + _NEXT), r"\1ou're"),
    (re.compile(r"\b([Tt])hey are\b" + _NEXT), r"\1hey're"),
    (re.compile(r"\b([Ss])he is\b" + _NEXT), r"\1he's"),
    (re.compile(r"\b([Hh])e is\b" + _NEXT), r"\1e's"),
    (re.compile(r"\b([Ww])e will\b" + _NEXT), r"\1e'll"),
    (re.compile(r"\bI will\b" + _NEXT), "I'll"),
    (re.compile(r"\b([Yy])ou will\b" + _NEXT), r"\1ou'll"),
    (re.compile(r"\b([Tt])hey will\b" + _NEXT), r"\1hey'll"),
    (re.compile(r"\b([Ii])t will\b" + _NEXT), _no_prep(r"\1t'll")),
    (re.compile(r"\b([Ww])e would\b" + _NEXT), r"\1e'd"),
    (re.compile(r"\bI would\b" + _NEXT), "I'd"),
    (re.compile(r"\b([Yy])ou would\b" + _NEXT), r"\1ou'd"),
    (re.compile(r"\b([Tt])hey would\b" + _NEXT), r"\1hey'd"),
]

# "have/has" contracts only as an auxiliary ("we've seen"), never for
# possession ("we have a model"). Gate on the next word looking like a past
# participle or an adverb that sits between auxiliary and participle.
_PARTICIPLES = set("""been seen done made built written given taken shown run
found got gotten come gone had kept left lost met put read said sent set
spent told thought brought become begun broken chosen drawn fallen felt heard
held known meant paid sold spoken stood understood won worn grown thrown
caught taught bought fought led fed dealt slept swept risen sworn torn woken
struck stuck sung drunk shrunk sprung lain""".split())
# Adverbs that sit between auxiliary and participle; the gate walks past
# them and tests the word that follows ("has only grown" yes, "have only
# one option" no).
_AUX_ADVERBS = set("already just never also now since yet long ever still only even".split())
_NOT_PARTICIPLES = {"open", "even", "often", "seven", "ten", "token", "speed",
                    "seed", "need", "red", "bed", "hundred", "screen", "green",
                    "sudden", "hidden", "garden", "golden", "kitchen", "children"}


def _aux_gate(suffix):
    def repl(m):
        tail, pos = m.string[m.end():m.end() + 60], 0
        for _ in range(4):
            nxt = re.match(r" +([A-Za-z]+)", tail[pos:])
            if not nxt:
                return None
            w = nxt.group(1).lower()
            if w in _AUX_ADVERBS or w.endswith("ly"):
                pos += nxt.end()
                continue
            if w not in _NOT_PARTICIPLES and (
                    w in _PARTICIPLES or w.endswith("ed") or w.endswith("en")):
                return m.group(1) + suffix
            return None
        return None
    return repl


RULES += [
    (re.compile(r"\b(I) have\b"), _aux_gate("'ve")),
    (re.compile(r"\b([Ww])e have\b"), _aux_gate("e've")),
    (re.compile(r"\b([Yy])ou have\b"), _aux_gate("ou've")),
    (re.compile(r"\b([Tt])hey have\b"), _aux_gate("hey've")),
    (re.compile(r"\b([Ii])t has\b"),
     lambda m: None if _prep_before(m) else _aux_gate("t's")(m)),
    (re.compile(r"\b([Tt])hat has\b"),
     lambda m: None if _prep_before(m) else _aux_gate("hat's")(m)),
    (re.compile(r"\b([Tt])here has\b"), _aux_gate("here's")),
    (re.compile(r"\b([Ss])he has\b"), _aux_gate("he's")),
    (re.compile(r"\b([Hh])e has\b"), _aux_gate("e's")),
]

# Directive bodies that are author prose and safe to convert. Anything else
# fenced (epigraph quotes, figures, tables, code) is protected.
PROSE_DIRECTIVES = {
    "admonition", "note", "tip", "warning", "caution", "important", "hint",
    "attention", "danger", "error", "seealso", "margin", "sidebar",
    "glossary", "dropdown",
}

FENCE_RE = re.compile(r"^(\s*)(`{3,})\s*(\{([a-z-]+)\})?")


def is_epigraph(text):
    """Plain-markdown epigraph cells: an italic quotation whose attribution
    line opens with an em-dash (the one sanctioned em-dash use in the book).
    These are real people's words and must stay verbatim."""
    lines = [l for l in text.split("\n") if l.strip()]
    return (lines and lines[0].lstrip().startswith("*")
            and any(l.lstrip().startswith("—") for l in lines))


def protected_mask(text):
    """Boolean list, True where text must not be rewritten."""
    if is_epigraph(text):
        return [True] * len(text)
    mask = [False] * len(text)

    def shield(a, b):
        for i in range(a, min(b, len(text))):
            mask[i] = True

    # 1. fenced blocks: protect delimiter lines always, bodies unless the
    #    innermost directive (and every enclosing one) is prose.
    pos = 0
    stack = []  # (fence_len, is_prose)
    for line in text.split("\n"):
        end = pos + len(line)
        m = FENCE_RE.match(line)
        if m:
            ticks = len(m.group(2))
            if stack and ticks >= stack[-1][0] and not m.group(3):
                stack.pop()
            else:
                stack.append((ticks, m.group(4) in PROSE_DIRECTIVES))
            shield(pos, end)
        elif stack and not all(p for _, p in stack):
            shield(pos, end)
        elif line.lstrip().startswith(">"):  # 2. blockquotes
            shield(pos, end)
        pos = end + 1

    # 3. inline code spans (within a line, so a span can never swallow a
    #    fenced directive body the way a multi-line match would)
    for m in re.finditer(r"(`+)[^`\n]*?\1", text):
        shield(*m.span())
    # 4. HTML comments
    for m in re.finditer(r"<!--[\s\S]*?-->", text):
        shield(*m.span())

    # 5. double-quoted spans (quote chars inside already-shielded code spans
    #    do not count). Straight quotes pair sequentially; an unmatched
    #    opener shields through to the end, erring on the safe side.
    straight = [i for i, ch in enumerate(text) if ch == '"' and not mask[i]]
    for j in range(0, len(straight) - 1, 2):
        shield(straight[j], straight[j + 1] + 1)
    if len(straight) % 2:
        shield(straight[-1], len(text))
    for m in re.finditer(r"“[\s\S]*?”", text):
        shield(*m.span())
    return mask


def convert(text, where, report):
    """Apply RULES outside protected spans; log each hit into report."""
    mask = protected_mask(text)
    hits = []  # (start, end, replacement)
    for pat, repl in RULES:
        for m in pat.finditer(text):
            a, b = m.span()
            new_text = repl(m) if callable(repl) else m.expand(repl)
            entry = (where, text[max(0, a - 35):a], m.group(0),
                     new_text or m.group(0), text[b:b + 35])
            ctx = text[max(0, a - 40):b + 40]
            if new_text is None or any(nv in ctx for nv in NEVER) or (
                    # coordination ellipsis: "what is and isn't feasible",
                    # "it is and will remain" keep the bare auxiliary
                    not new_text.endswith("n't")
                    and re.match(r" +(and|or|but|nor)\b", text[b:])):
                report["guarded"].append(entry)
            elif any(mask[a:b]):
                report["skipped"].append(entry)
            else:
                hits.append((a, b, new_text))
                report["applied"].append(entry)
    out, last = [], 0
    for a, b, r in sorted(hits):
        out.append(text[last:a]); out.append(r); last = b
    out.append(text[last:])
    return "".join(out)


def resplit(s, original):
    """Rebuild a notebook source list; fall back safely if shape differs."""
    parts = s.split("\n")
    lines = [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
    assert "".join(lines) == s
    return lines


def process_notebook(path, dry, report):
    raw = Path(path).read_text(encoding="utf-8")
    nb = json.loads(raw)
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell["source"]
        joined = "".join(src) if isinstance(src, list) else src
        new = convert(joined, f"{Path(path).name} cell {i}", report)
        if new != joined:
            cell["source"] = resplit(new, src) if isinstance(src, list) else new
    # Preserve the file's JSON style (see trim_outputs.py): unicode escaping
    # detected via isascii, trailing newline kept, no write when unchanged.
    out = json.dumps(nb, indent=1, ensure_ascii=raw.isascii())
    if raw.endswith("\n"):
        out += "\n"
    if out != raw and not dry:
        Path(path).write_text(out, encoding="utf-8")
    return out != raw


def process_md(path, dry, report):
    raw = Path(path).read_text(encoding="utf-8")
    new = convert(raw, Path(path).name, report)
    if new != raw and not dry:
        Path(path).write_text(new, encoding="utf-8")
    return new != raw


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry = "--dry-run" in sys.argv
    files = args or DEFAULT_FILES
    report = {"applied": [], "skipped": [], "guarded": []}
    changed = []
    for f in files:
        fn = process_md if f.endswith(".md") else process_notebook
        if fn(f, dry, report):
            changed.append(f)
    for tag, rows in (("GUARD", report["guarded"]), ("SKIP", report["skipped"]),
                      ("EDIT", report["applied"])):
        for where, before, old, newt, after in rows:
            ctx = (before + "[" + old + " -> " + newt + "]" + after).replace("\n", "\\n")
            print(f"{tag}  [{where}] ...{ctx}...")
    print(f"\n{len(report['applied'])} applied, {len(report['skipped'])} protected, "
          f"{len(report['guarded'])} guard-rejected, "
          f"{len(changed)} files {'would change' if dry else 'changed'}.")
