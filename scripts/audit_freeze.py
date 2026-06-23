"""
audit_freeze.py
Find code cells whose saved OUTPUT is nondeterministic and should carry a
"freeze" tag (see scripts/freeze_cell.py), so execute_notebook.py re-runs don't
drift. Reports per chapter and prints ready-to-run freeze commands.

Usage: python scripts/audit_freeze.py                 # all chapters
       python scripts/audit_freeze.py chapters/foo.ipynb [bar.ipynb ...]

A cell needs freezing when it (a) has non-empty saved output and (b) is
nondeterministic:
  - LLM sampling: ask/_ask/chat/_chat/agent/rag/ask_image(s)/.chat(/.generate(
    and local wrappers that (transitively) call them — resolved to a fixpoint,
    so route, clarify, ask_with_confidence, grounded_ask, safe_answer, ... all
    count, UNLESS pinned with temperature:0 / num_predict:0.
  - Wall-clock timing: time_call / perf_counter / time.time / %timeit.
  - Unseeded randomness: np.random / random. with NO seed. default_rng(x) for any
    arg is seeded; so is np.random.seed(...) / RandomState(...) / seed=.
Deterministic (never flagged): tokenizers, embeddings, federated_avg (np.mean),
seeded RNG.

SAFETY: a frozen cell is SKIPPED entirely on re-run, so it must not provide a
name a still-live cell needs. Dependencies are resolved with symtable (real
scope: a `def f(prompt)` parameter is NOT a cross-cell dependency). A fixpoint
demotes any candidate that would orphan a live cell to UNSAFE — those need a
code change (split the deterministic definition out, or pin temperature:0), not
a tag.
"""
import ast
import json
import re
import sys
import glob
import symtable
from pathlib import Path

_CHAPTERS = str(Path(__file__).parent.parent / "chapters" / "*.ipynb")

# "Inherent" nondeterminism: LLM sampling and wall-clock timing. These cannot be
# seeded away, so any unpinned call makes the cell's output nonreproducible.
INHERENT = [
    r"\bask\s*\(", r"\b_ask\s*\(", r"\bask_image\s*\(", r"\bask_images\s*\(",
    r"\bchat\s*\(", r"\b_chat\s*\(", r"\.chat\s*\(", r"\.generate\s*\(",
    r"\bagent\s*\(", r"\brag\s*\(", r"\.ask\s*\(",
    r"\bgrade_is_prime\s*\(", r"\bprompt_injection_demo\s*\(",
    r"\bpoisoning_demo\s*\(",
    r"\btime_call\s*\(", r"\bcompare_models\s*\(",
    r"\badd_noise\s*\(", r"\bprivate_avg\s*\(",  # global np.random.laplace
    r"time\.time\s*\(", r"perf_counter\s*\(", r"%%?timeit",
]
INHERENT_RE = re.compile("|".join(INHERENT))
# LLM sampling counts as deterministic only when pinned to greedy / a 0-token counter.
PIN_RE = re.compile(r'temperature["\']?\s*[:=]\s*0(?:\.0)?\b|num_predict["\']?\s*[:=]\s*0\b')
# Bare RNG is nondeterministic only when not seeded.
RNG_RE = re.compile(r"np\.random|\brandom\.")
# default_rng(x) is seeded for ANY explicit arg; only bare default_rng() draws
# fresh OS entropy. seed=<x> / np.random.seed(...) / RandomState(...) also seed.
SEED_RE = re.compile(
    r"default_rng\s*\(\s*[^)\s]|np\.random\.seed\s*\(|RandomState\s*\(|manual_seed\s*\(|seed\s*=\s*\w"
)


def assigned_and_loaded(src):
    """Return (names this cell provides, names it needs from a prior cell).

    Uses symtable for real scope resolution: a function parameter `prompt` is
    bound inside its function, NOT a cross-cell dependency, whereas a module-level
    reference to an unassigned name (or a global used inside a function) is.
    """
    try:
        top = symtable.symtable(src, "<cell>", "exec")
    except SyntaxError:
        return set(), set()
    assigned, free = set(), set()
    for s in top.get_symbols():
        if s.is_assigned() or s.is_imported():
            assigned.add(s.get_name())        # provided to later cells
        elif s.is_referenced():
            free.add(s.get_name())            # must come from an earlier cell
    return assigned, free


def has_output(cell):
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream" and "".join(o.get("text", [])).strip():
            return True
        if o.get("output_type") in ("execute_result", "display_data") and o.get("data"):
            return True
        if o.get("output_type") == "error":
            return True
    return False


def nd_local_funcs(cells):
    """Names of locally-defined functions that are nondeterministic because their
    body invokes an nd helper — either a built-in nd call (ask/_ask/chat/_chat/
    timing/...) or, transitively, another local nd function. Resolved to a
    fixpoint, so a chain like safe_answer -> ask_with_confidence -> _ask is caught
    in full and calls to ANY link count as nd."""
    defs = []  # (name, body_src) for every def found in any code cell
    for c in cells:
        src = "".join(c["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append((node.name, ast.get_source_segment(src, node) or ""))

    # Seed: functions that directly call a built-in nd helper or use unseeded RNG.
    names = set()
    for name, body in defs:
        rng = RNG_RE.search(body) and not SEED_RE.search(body)
        if INHERENT_RE.search(body) or rng:
            names.add(name)

    # Fixpoint: a function that calls an already-known nd local function is nd too.
    changed = True
    while changed:
        changed = False
        wrap_re = re.compile("|".join(rf"\b{re.escape(n)}\s*\(" for n in names)) if names else None
        for name, body in defs:
            if name not in names and wrap_re and wrap_re.search(body):
                names.add(name)
                changed = True
    return names


def audit(path):
    """Return (info, rows): info = per-cell facts; rows = [(candidate, clashes)],
    clashes empty => SAFE to freeze."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    local_nd = nd_local_funcs(cells)
    local_re = re.compile("|".join(rf"\b{re.escape(n)}\s*\(" for n in local_nd)) if local_nd else None

    info = []
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        inherent = bool(INHERENT_RE.search(src) or (local_re and local_re.search(src)))
        rng = bool(RNG_RE.search(src))
        nd = (inherent and not PIN_RE.search(src)) or (rng and not SEED_RE.search(src))
        a, f = assigned_and_loaded(src)
        info.append(dict(i=i, id=c.get("id"), frozen="freeze" in c.get("metadata", {}).get("tags", []),
                         nd=nd, out=has_output(c), assigned=a, free=f, src=src))

    candidates = [x for x in info if x["nd"] and x["out"] and not x["frozen"]]
    already = {x["i"] for x in info if x["frozen"]}

    # Fixpoint: a frozen cell must not provide a free-load needed by any cell we
    # are NOT freezing. Demote unsafe candidates and repeat until stable, so a
    # demoted cell can in turn make its providers unsafe.
    freeze = {x["i"] for x in candidates}
    reasons = {}
    changed = True
    while changed:
        changed = False
        for x in candidates:
            if x["i"] not in freeze:
                continue
            clashes = [(y["id"], sorted(x["assigned"] & y["free"]))
                       for y in info
                       if y["i"] > x["i"] and y["i"] not in freeze and y["i"] not in already
                       and (x["assigned"] & y["free"])]
            if clashes:
                freeze.discard(x["i"])
                reasons[x["i"]] = clashes
                changed = True

    return info, [(x, reasons.get(x["i"], [])) for x in candidates]


def _first_code_line(src):
    for line in src.strip().splitlines():
        if line.strip() and not line.startswith("#"):
            return line
    return src.strip().splitlines()[0] if src.strip() else ""


def main(paths):
    commands = {}
    for path in paths:
        info, rows = audit(path)
        safe = [x for x, clash in rows if not clash]
        unsafe = [(x, clash) for x, clash in rows if clash]
        nfroz = sum(x["frozen"] for x in info)
        print(f"\n=== {Path(path).name}  ({len(info)} code, {nfroz} frozen) ===")
        print(f"  nondeterministic + output, not yet frozen: {len(rows)}"
              f"   SAFE={len(safe)}  UNSAFE={len(unsafe)}")
        for x in safe:
            print(f"    SAFE   {x['id']:<20} | {_first_code_line(x['src'])[:60]}")
        for x, clash in unsafe:
            print(f"    UNSAFE {x['id']:<20} | live cells need: {clash}")
        if safe:
            commands[path] = [x["id"] for x in safe]

    if commands:
        print("\n--- to freeze the SAFE cells ---")
        for path, ids in commands.items():
            print(f"uv run python scripts/freeze_cell.py {path} " + " ".join(ids))
    else:
        print("\nAll chapters covered — no SAFE candidates to freeze.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    paths = []
    for a in (args or [_CHAPTERS]):
        paths.extend(sorted(glob.glob(a)) if "*" in a else [a])
    main(paths)
